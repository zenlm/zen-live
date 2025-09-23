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

load_dotenv()

os.environ["MODE"] = "UI"
cur_dir = Path(__file__).parent

API_KEY = os.environ['API_KEY']  # Set with: export DASHSCOPE_API_KEY=xxx
API_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen3-livetranslate-flash-realtime"
VOICES = ["Cherry", "Nofish", "Jada", "Dylan", "Sunny", "Peter", "Kiki", "Eric"]

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

    def copy(self):
        return LiveTranslateHandler()

    @staticmethod
    def msg_id() -> str:
        return f"event_{secrets.token_hex(10)}"

    async def start_up(self):
        try:
            await self.wait_for_args()
            args = self.latest_args
            src_language_name = args[2] if len(args) > 2 else "English" # 现在 dropdown 返回的是全称
            target_language_name = args[3] if len(args) > 3 else "Chinese" 
            src_language_code = LANG_MAP_REVERSE[src_language_name]
            target_language_code = LANG_MAP_REVERSE[target_language_name]

            # src_language = args[2] if len(args) > 2 else "zh"  # 新增源语言参数
            # target_language = args[3] if len(args) > 3 else "en" 
            voice_id = args[4] if len(args) > 4 else "Cherry"
            
            if src_language_code == target_language_code:
                print(f"⚠️ 源语言和目标语言相同({target_language_name})，将以复述模式运行")

            async with connect(API_URL, additional_headers=headers) as conn:
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

                    # elif event_type in ("response.text.text", "response.audio_transcript.text"):
                    #     # 中间结果 + stash（stash通常是句子完整缓存）
                    #     stash_text = event.get("stash", "")
                    #     text_field = event.get("text", "")
                    #     if stash_text or text_field:
                    #         await self.output_queue.put(
                    #             AdditionalOutputs({"role": "assistant", "content": stash_text or text_field})
                    #         )

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

    async def emit(self) -> tuple[int, np.ndarray] | AdditionalOutputs | None:
        return await wait_for_item(self.output_queue)

    async def shutdown(self) -> None:
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
latest_message = gr.Textbox(type="text", visible=False)

# 可选：暂时禁用 TURN 配置进行测试
rtc_config = get_cloudflare_turn_credentials_async if get_space() else None
# rtc_config = None  # 取消注释可禁用 TURN 测试

stream = Stream(
    LiveTranslateHandler(),
    mode="send-receive",
    modality="audio",
    additional_inputs=[src_language, language, voice, chatbot],  # 添加 src_language
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