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


class LiveTranslateHandler(AsyncAudioVideoStreamHandler):
    def __init__(self) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=24_000,
            input_sample_rate=16_000,
        )
        self.connection = None
        self.output_queue = asyncio.Queue()
        self.video_queue = asyncio.Queue()

        self.last_send_time = 0.0     # 上次发送时间
        self.video_interval = 0.5     # 间隔 0.5 s
        self.latest_frame = None

        self.awaiting_new_message = True
        self.stable_text = ""  # 黑色部分
        self.temp_text = ""    # 灰色部分


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

                    if event_type in ("response.text.text", "response.audio_transcript.text"):
                        # 更新稳定部分（stash / text 认为是已确认的）
                        self.stable_text = event.get("text", "") or ""
                        self.temp_text = event.get("stash", "") or ""
                        # self.stable_text = event.get("stash", "") or ""
                        # self.temp_text = event.get("text", "") or ""

                        print(f"[STABLE] {self.stable_text}")
                        print(f"[TEMP] {self.temp_text}")
                        await self.output_queue.put(
                            AdditionalOutputs({
                                "role": "assistant",
                                # 将稳定文本变黑色、临时文本变灰色
                                "content": f"<span style='color:black'>{self.stable_text}</span>"
                                        f"<span style='color:gray'>{self.temp_text}</span>",
                                "update": True,
                                "new_message": self.awaiting_new_message
                            })
                        )
                        self.awaiting_new_message = False

                    elif event_type == "response.audio_transcript.done":
                        transcript = event.get("transcript", "")
                        print(f"[DONE] {transcript}")
                        if transcript:
                            self.stable_text = transcript
                            self.temp_text = ""
                            await self.output_queue.put(
                                AdditionalOutputs({
                                    "role": "assistant",
                                    "content": f"<span style='color:black'>{self.stable_text}</span>",
                                    "update": True,
                                    "new_message": self.awaiting_new_message
                                })
                            )
                        # 开启新气泡
                        self.awaiting_new_message = True
                        self.stable_text = ""
                        self.temp_text = ""

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

    # 客户端 to 服务端
    async def video_receive(self, frame: np.ndarray):
        self.latest_frame = frame
        if self.connection is None:
            return

        # Push frame to local queue for display immediately
        await self.video_queue.put(frame)  

        now = time.time()
        if now - self.last_send_time < self.video_interval:
            return
        self.last_send_time = now

        # 发送到云端
        frame_resized = cv2.resize(frame, (640, 360))
        _, buf = cv2.imencode(".jpg", frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        img_b64 = base64.b64encode(buf.tobytes()).decode()

        await self.connection.send(
            json.dumps(
                {
                    "event_id": self.msg_id(),
                    "type": "input_image_buffer.append",
                    "image": img_b64,
                }
            )
        )

    # 服务端 2 客户端
    async def video_emit(self):
        # frame = await wait_for_item(self.video_queue, 0.01)
        # # print(f"能够显示出来的=================={frame.dtype}==================")
        # if frame is not None:
        #     return frame
        # else:
        #     # return np.zeros((100, 100, 3), dtype=np.uint8)
        #     return None
        if self.latest_frame is not None:
            return self.latest_frame
        return np.zeros((100, 100, 3), dtype=np.uint8)


    async def receive(self, frame):
        if self.connection is None:
            return
        sr, array = frame          # frame 一定是 (sr, np.ndarray)
        array = array.squeeze()
        audio_b64 = base64.b64encode(array.tobytes()).decode()
        await self.connection.send(
            json.dumps(
                {
                    "event_id": self.msg_id(),
                    "type": "input_audio_buffer.append",
                    "audio": audio_b64,
                }
            )
        )


    async def emit(self) -> tuple[int, np.ndarray] | AdditionalOutputs | None:
        return await wait_for_item(self.output_queue)

    async def shutdown(self) -> None:
        """关闭连接并清理资源"""
        # if self.video_capture:
        #     self.video_capture.release()  # 释放视频设备
        #     self.video_capture = None

        if self.connection:
            await self.connection.close()
            self.connection = None

        # 清空队列
        while not self.output_queue.empty():
            self.output_queue.get_nowait()


def update_chatbot(chatbot: list[dict], response: dict):
    is_update = response.pop("update", False)
    new_message_flag = response.pop("new_message", False)
    stable_html = response["content"]

    if is_update:
        if new_message_flag or not chatbot:
            chatbot.append({"role": "assistant", "content": stable_html})
        else:
            if chatbot[-1]["role"] == "assistant":
                chatbot[-1]["content"] = stable_html
            else:
                chatbot.append({"role": "assistant", "content": stable_html})
    else:
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
# video_flag = gr.Dropdown(
#     choices=["True", "False"],
#     value="False",
#     label="Use Video"
# )

latest_message = gr.Textbox(type="text", visible=False)

# 可选：暂时禁用 TURN 配置进行测试
rtc_config = get_cloudflare_turn_credentials_async if get_space() else None
# rtc_config = None  # 取消注释可禁用 TURN 测试


stream = Stream(
    LiveTranslateHandler(),
    mode="send-receive",
    modality="audio-video",
    # modality="audio",
    additional_inputs=[src_language, language, voice, chatbot],
    additional_outputs=[chatbot],
    additional_outputs_handler=update_chatbot,
    rtc_configuration=rtc_config,
    concurrency_limit=5 if get_space() else None,
    time_limit=90 if get_space() else None,
)

#  前端
def enhance_ui():
    with stream.ui as demo:
        gr.HTML("""
    <style>
        .gradio-container .wrap {
            display: flex;
            flex-direction: column;
            align-items: center;
          }
        .gradio-container video{
            width: 100%;
            max-width: 500px;
            max-height: 500px;
            aspect-ratio: 1/1;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            margin: 7% 10px;
        }

        .button-wrap.svelte-g69fcx.svelte-g69fcx {
            position: absolute;
            background-color: color-mix(in srgb,var(--block-background-fill) 50%,transparent);
            border: 1px solid var(--border-color-primary);
            padding: var(--size-1-5);
            display: flex;
            bottom: 300px;
            left: 50%;
            transform: translate(-50%);
            box-shadow: var(--shadow-drop-lg);
            border-radius: var(--radius-xl);
            line-height: var(--size-3);
            color: var(--button-secondary-text-color)
        }
        label.svelte-j0zqjt.svelte-j0zqjt {
            display: flex;
            align-items: center;
            z-index: var(--layer-2);
            box-shadow: var(--block-label-shadow);
            border: var(--block-label-border-width) solid var(--block-label-border-color);
            border-top: none;
            border-left: none;
            border-radius: var(--block-label-radius);
            background: var(--block-label-background-fill);
            padding: var(--block-label-padding);
            pointer-events: none;
            color: var(--block-label-text-color);
            font-weight: var(--block-label-text-weight);
            font-size: var(--block-label-text-size);
            line-height: var(--line-sm)
        }
    </style>
        """)
    return demo

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
    exit(0)
    # 可扩展为执行更多清理逻辑


signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

if __name__ == "__main__":
    import os

    if (mode := os.getenv("MODE")) == "UI":
        demo = enhance_ui()
        demo.launch()
        # stream.ui.launch(server_port=7862)
    elif mode == "PHONE":
        stream.fastphone(host="0.0.0.0")
    else:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0")