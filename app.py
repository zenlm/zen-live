"""Simultaneous speech translation over FastRTC using DashScope Qwen3 LiveTranslate.
 - Streams mic audio (16k PCM16) to DashScope Realtime
 - Receives translated text deltas and 24k PCM16 TTS audio
 - Plays audio via FastRTC and shows text in a Gradio Chatbot
Set DASHSCOPE_API_KEY in the environment before running.
"""
import os
import time
import base64
import asyncio
import json
import secrets
import signal
from pathlib import Path

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastrtc import (
    AdditionalOutputs,
    AsyncStreamHandler,
    Stream,
    get_cloudflare_turn_credentials_async,
    wait_for_item,
)
from gradio.utils import get_space
from websockets.asyncio.client import connect
import ssl
import certifi

import cv2

load_dotenv()

os.environ["MODE"] = "UI"
cur_dir = Path(__file__).parent

API_KEY = os.environ['API_KEY']  # Set with: export DASHSCOPE_API_KEY=xxx
API_URL = "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime?model=qwen3-livetranslate-flash-realtime"
VOICES = ["Cherry", "Nofish", "Jada", "Dylan", "Sunny", "Peter", "Kiki", "Eric"]

ssl_context = ssl.create_default_context(cafile=certifi.where())
# ssl_context = ssl._create_unverified_context()  # 禁用证书验证

if not API_KEY:
    raise RuntimeError("Missing DASHSCOPE_API_KEY environment variable.")
headers = {"Authorization": "Bearer " + API_KEY}
LANG_MAP = {
    "en": "English",
    "zh": "Chinese",
    "ru": "Russian",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
    "ko": "Korean",
    "ja": "Japanese",
    "yue": "Cantonese",
    "id": "Indonesian",
    "vi": "Vietnamese",
    "th": "Thai",
    "ar": "Arabic",
    "hi": "Hindi",
    "el": "Greek",
    "tr": "Turkish"
}
LANG_MAP_REVERSE = {v: k for k, v in LANG_MAP.items()}
# SRC_LANGUAGES = ["en", "zh", "ru", "fr", "de", "pt", "es", "it", "ko", "ja", "yue", "id", "vi", "th", "ar", "hi", "el", "tr"]  # 使用相同的语言列表
# TARGET_LANGUAGES = ["en", "zh", "ru", "fr", "de", "pt", "es", "it", "ko", "ja", "yue", "id", "vi", "th", "ar"]

SRC_LANGUAGES = [LANG_MAP[code] for code in ["en", "zh", "ru", "fr", "de", "pt", "es", "it", "ko", "ja", "yue", "id", "vi", "th", "ar", "hi", "el", "tr"]]
TARGET_LANGUAGES = [LANG_MAP[code] for code in ["en", "zh", "ru", "fr", "de", "pt", "es", "it", "ko", "ja", "yue", "id", "vi", "th", "ar"]]


class LiveTranslateHandler(AsyncStreamHandler):
    def __init__(self) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=24_000,
            input_sample_rate=16_000,
        )
        self.connection = None
        self.output_queue = asyncio.Queue()
        self.video_capture = None  # 视频捕获设备
        self.last_capture_time = 0  # 上次视频帧捕获时间戳
        self.enable_video = False  

    def setup_video(self):
        """设置视频捕获设备"""
        self.video_capture = cv2.VideoCapture(0)  # 打开默认摄像头
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # 设置宽度
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 设置高度
        self.video_capture.set(cv2.CAP_PROP_FPS, 30)  # 设置 FPS

    def get_video_frame(self) -> bytes | None:
        """获取视频帧并处理成缩放后的字节"""
        if not self.video_capture:
            return None

        # 获取当前时间
        current_time = time.time()

        # 每隔 0.5 秒截取一帧
        if current_time - self.last_capture_time >= 0.5:
            self.last_capture_time = current_time
            ret, frame = self.video_capture.read()  # 捕获当前帧
            if ret:
                # 压缩并调整分辨率
                resized_frame = cv2.resize(frame, (640, 360))  # 确保分辨率低于 480p
                # 使用 JPEG 格式编码视频帧
                _, encoded_image = cv2.imencode('.jpg', resized_frame)
                return encoded_image.tobytes()
        return None

    async def send_image_frame(self, image_bytes: bytes, *, event_id: str | None = None):
        """将图像数据发送给服务器"""
        if not self.connection:
            return

        if not image_bytes:
            raise ValueError("image_bytes 不能为空")

        # 编码为 Base64
        image_b64 = base64.b64encode(image_bytes).decode()

        event = {
            "event_id": event_id or self.msg_id(),
            "type": "input_image_buffer.append",
            "image": image_b64,
        }

        await self.connection.send(json.dumps(event))


    def copy(self):
        return LiveTranslateHandler()

    @staticmethod
    def msg_id() -> str:
        return f"event_{secrets.token_hex(10)}"

    async def start_up(self):
        try:
            await self.wait_for_args()
            args = self.latest_args
            src_language_name = args[2] if len(args) > 2 else "Chinese" # 现在 dropdown 返回的是全称
            target_language_name = args[3] if  len(args) > 3 else "English" 
            src_language_code = LANG_MAP_REVERSE[src_language_name]
            target_language_code = LANG_MAP_REVERSE[target_language_name]

            voice_id = args[4] if len(args) > 4 else "Cherry"

            self.enable_video = True if args[5] == "True" else False

            if self.enable_video:
                self.setup_video()  # 初始化视频设备
            
            if src_language_code == target_language_code:
                print(f"⚠️ 源语言和目标语言相同({target_language_name})，将以复述模式运行")

            async with connect(API_URL, additional_headers=headers, ssl=ssl_context) as conn:
                self.client = conn
                await conn.send(
                    json.dumps(
                        {
                            "event_id": self.msg_id(),
                            "type": "session.update", 
                            "session": {
                                "modalities": ["text", "audio"],
                                "voice": voice_id,
                                "input_audio_format": "pcm16",
                                "output_audio_format": "pcm16",
                                "translation": {
                                    "source_language": src_language_code,  # 添加源语言
                                    "language": target_language_code
                                }
                            },
                        }
                    )
                )
                self.connection = conn

                # WebSocket 收到的每一个响应（data）是一个 JSON 事件，表示翻译任务的进展。
                async for data in self.connection:
                    event = json.loads(data)
                    if "type" not in event:
                        continue
                    event_type = event["type"]

                    if event_type == "response.audio_transcript.delta":
                        # 增量字幕
                        text = event.get("transcript", "")
                        if text:
                            await self.output_queue.put(
                                AdditionalOutputs({"role": "assistant", "content": text})
                            )

                    elif event_type == "response.audio_transcript.done":
                        # 最终完整句子
                        transcript = event.get("transcript", "")
                        if transcript:
                            await self.output_queue.put(
                                AdditionalOutputs({"role": "assistant", "content": transcript})
                            )

                    elif event_type == "response.audio.delta":
                        audio_b64 = event.get("delta", "")
                        if audio_b64:
                            audio_data = base64.b64decode(audio_b64)
                            audio_array = np.frombuffer(audio_data, dtype=np.int16).reshape(1, -1)
                            await self.output_queue.put(
                                (self.output_sample_rate, audio_array)
                            )

        except Exception as e:
            print(f"Connection error: {e}")
            await self.shutdown()

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        if not self.connection:
            return
        _, array = frame
        array = array.squeeze()
        audio_message = base64.b64encode(array.tobytes()).decode("utf-8")
        await self.connection.send(
            json.dumps(
                {
                    "event_id": self.msg_id(),
                    "type": "input_audio_buffer.append",
                    "audio": audio_message,
                }
            )
        )

        # 视频部分
        if self.enable_video:
            image_frame = self.get_video_frame()
            if image_frame:
                await self.send_image_frame(image_frame)


    async def emit(self) -> tuple[int, np.ndarray] | AdditionalOutputs | None:
        return await wait_for_item(self.output_queue)

    async def shutdown(self) -> None:
        """关闭连接并清理资源"""
        if self.video_capture:
            self.video_capture.release()  # 释放视频设备
            self.video_capture = None

        if self.connection:
            await self.connection.close()
            self.connection = None

        # 清空队列
        while not self.output_queue.empty():
            self.output_queue.get_nowait()


def update_chatbot(chatbot: list[dict], response: dict):
    chatbot.append(response)
    return chatbot


chatbot = gr.Chatbot(type="messages")
src_language = gr.Dropdown(
    choices=SRC_LANGUAGES,
    value="English",   # 改成全称
    type="value",
    label="Source Language"
)
language = gr.Dropdown(
    choices=TARGET_LANGUAGES,
    value="Chinese",   # 改成全称
    type="value",
    label="Target Language"
)
voice = gr.Dropdown(choices=VOICES, value=VOICES[0], type="value", label="Voice")
video_flag = gr.Dropdown(
    choices=["True", "False"],
    value="False",
    label="Use Video"
)

latest_message = gr.Textbox(type="text", visible=False)

# 可选：暂时禁用 TURN 配置进行测试
rtc_config = get_cloudflare_turn_credentials_async if get_space() else None
# rtc_config = None  # 取消注释可禁用 TURN 测试


stream = Stream(
    LiveTranslateHandler(),
    mode="send-receive",
    modality="audio",
    additional_inputs=[src_language, language, voice, video_flag,chatbot],
    additional_outputs=[chatbot],
    additional_outputs_handler=update_chatbot,
    rtc_configuration=rtc_config,
    concurrency_limit=5 if get_space() else None,
    time_limit=90 if get_space() else None,
)



app = FastAPI()

stream.mount(app)


@app.get("/")
async def _():
    rtc_config = await get_cloudflare_turn_credentials_async() if get_space() else None
    html_content = (cur_dir / "index.html").read_text()
    html_content = html_content.replace("__RTC_CONFIGURATION__", json.dumps(rtc_config))
    return HTMLResponse(content=html_content)


@app.get("/outputs")
def _(webrtc_id: str):
    async def output_stream():
        import json

        async for output in stream.output_stream(webrtc_id):
            s = json.dumps(output.args[0])
            yield f"event: output\ndata: {s}\n\n"

    return StreamingResponse(output_stream(), media_type="text/event-stream")


def handle_exit(sig, frame):
    print("Shutting down gracefully...")
    # 可扩展为执行更多清理逻辑


signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

if __name__ == "__main__":
    import os

    if (mode := os.getenv("MODE")) == "UI":
        stream.ui.launch(server_port=7860)
    elif mode == "PHONE":
        stream.fastphone(host="0.0.0.0", port=7860)
    else:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=7860)