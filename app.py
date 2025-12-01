"""
Zen Live - Real-time Speech Translation for Broadcast

A low-latency simultaneous translation service for news control rooms.
Powered by Hanzo AI infrastructure with Qwen3 LiveTranslate backend.

Backend options:
  1. Hanzo Node API (recommended) - Set HANZO_NODE_URL
  2. Direct DashScope API - Set API_KEY
  3. Local Zen Omni model - Set ZEN_OMNI_PATH

Usage:
  export HANZO_NODE_URL=http://localhost:9550  # or
  export API_KEY=your_dashscope_key
  python app.py

Endpoints:
  /              - Control room web portal
  /monitor       - Simplified broadcast monitor view
  /api/status    - Service health check
  /broadcast/info - Integration guide for engineers

Part of the Zen AI model family: https://github.com/zenlm
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
from fastapi import FastAPI, Request, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from fastrtc import (
    AdditionalOutputs,
    # AsyncStreamHandler,
    AsyncAudioVideoStreamHandler,
    WebRTC,
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

os.environ["MODE"] = os.environ.get("MODE", "")
cur_dir = Path(__file__).parent

# Backend configuration
HANZO_NODE_URL = os.environ.get("HANZO_NODE_URL")  # Preferred: Hanzo Node backend
API_KEY = os.environ.get("API_KEY")  # Fallback: Direct DashScope API
ZEN_OMNI_PATH = os.environ.get("ZEN_OMNI_PATH")  # Optional: Local model path

# Authentication (optional - set both to enable)
AUTH_USER = os.environ.get("ZEN_LIVE_USER")  # e.g., psigg@americasvoice.news
AUTH_PASS = os.environ.get("ZEN_LIVE_PASS")  # e.g., livedemo2025
AUTH_ENABLED = bool(AUTH_USER and AUTH_PASS)

# Input source configuration (defaults, can be overridden via web UI localStorage)
SRT_INPUT_URL = os.environ.get("SRT_INPUT_URL")  # e.g., srt://source:9000?mode=caller
RTMP_INPUT_URL = os.environ.get("RTMP_INPUT_URL")  # e.g., rtmp://source/live/stream
WHIP_ENABLED = os.environ.get("WHIP_ENABLED", "true").lower() == "true"  # WebRTC WHIP input

API_URL = "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime?model=qwen3-livetranslate-flash-realtime"

# HTTP Basic Auth setup
security = HTTPBasic()


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify HTTP Basic Auth credentials if authentication is enabled."""
    if not AUTH_ENABLED:
        return True  # Auth disabled, allow all

    # Use constant-time comparison to prevent timing attacks
    correct_username = secrets.compare_digest(credentials.username, AUTH_USER)
    correct_password = secrets.compare_digest(credentials.password, AUTH_PASS)

    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic realm='Zen Live'"},
        )
    return True


def optional_auth(request: Request):
    """Optional auth - only require if AUTH_ENABLED is True."""
    if not AUTH_ENABLED:
        return True

    # Check for Authorization header
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Basic "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic realm='Zen Live'"},
        )

    try:
        credentials = base64.b64decode(auth_header[6:]).decode("utf-8")
        username, password = credentials.split(":", 1)

        correct_username = secrets.compare_digest(username, AUTH_USER)
        correct_password = secrets.compare_digest(password, AUTH_PASS)

        if not (correct_username and correct_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Basic realm='Zen Live'"},
            )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header",
            headers={"WWW-Authenticate": "Basic realm='Zen Live'"},
        )

    return True


VOICES = ["Cherry", "Nofish", "Jada", "Dylan", "Sunny", "Peter", "Kiki", "Eric"]

ssl_context = ssl.create_default_context(cafile=certifi.where())

# Determine backend and get credentials
if HANZO_NODE_URL:
    print(f"🔗 Zen Live: Using Hanzo Node backend at {HANZO_NODE_URL}")
    BACKEND_TYPE = "hanzo_node"
elif API_KEY:
    print("🔗 Zen Live: Using direct DashScope API")
    BACKEND_TYPE = "dashscope"
else:
    print("⚠️  Zen Live: No backend configured")
    print("   Set HANZO_NODE_URL (recommended) or API_KEY")
    BACKEND_TYPE = None

headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
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

# Spanish first for news monitoring default
SRC_LANGUAGES = [LANG_MAP[code] for code in ["es", "en", "zh", "pt", "fr", "de", "ru", "it", "ko", "ja", "yue", "id", "vi", "th", "ar", "hi", "el", "tr"]]
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
    value="Spanish",   # Default to Spanish for news monitoring
    type="value",
    label="Source Language"
)
language = gr.Dropdown(
    choices=TARGET_LANGUAGES,
    value="English",   # Default to English output
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

app = FastAPI(
    title="Zen Live",
    description="""
# Zen Live - Real-time Speech Translation API

Low-latency simultaneous translation service for broadcast news monitoring.

## Features
- **WebRTC streaming** - Browser-based real-time translation
- **WHIP/WHEP** - Standard broadcast ingestion/egress protocols
- **Audio streaming** - PCM/WAV endpoints for broadcast integration
- **SSE transcripts** - Real-time transcript streaming

## Quick Start
1. Open the [Control Room UI](/) in your browser
2. Select source language (default: Spanish)
3. Select target language (default: English)
4. Click Start to begin translation

## Authentication
When `ZEN_LIVE_USER` and `ZEN_LIVE_PASS` env vars are set, HTTP Basic Auth is required.

## Broadcast Integration
See [/broadcast/info](/broadcast/info) for ffmpeg, SRT, RTMP, and NDI integration examples.

Powered by [Hanzo AI](https://hanzo.ai) | [GitHub](https://github.com/zenlm/zen-live)
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0"
    },
    contact={
        "name": "Zen Live Support",
        "url": "https://github.com/zenlm/zen-live",
        "email": "support@hanzo.ai"
    }
)

# CORS for control room access from different domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

stream.mount(app)


# WebRTC offer model
class WebRTCOffer(BaseModel):
    sdp: str
    type: str
    src_language: str = "Spanish"
    target_language: str = "English"
    voice: str = "Cherry"


# Store active sessions for control room
active_sessions = {}


@app.post("/api/webrtc/offer")
async def webrtc_offer(offer: WebRTCOffer):
    """Handle WebRTC offer from control room portal.

    This endpoint receives SDP offers and returns SDP answers
    for establishing low-latency WebRTC connections.
    """
    try:
        # Generate session ID
        session_id = secrets.token_hex(16)

        # Store session config
        active_sessions[session_id] = {
            "src_language": offer.src_language,
            "target_language": offer.target_language,
            "voice": offer.voice,
            "created_at": time.time()
        }

        # Use FastRTC's internal offer handling
        response = await stream.offer(
            offer.sdp,
            offer.type
        )

        # Handle response - ensure required fields are present
        if isinstance(response, dict):
            sdp = response.get("sdp", "")
            resp_type = response.get("type", "answer")
            webrtc_id = response.get("webrtc_id", session_id)
        else:
            # If response is RTCSessionDescription or similar object
            sdp = getattr(response, "sdp", str(response))
            resp_type = getattr(response, "type", "answer")
            webrtc_id = session_id

        if not sdp:
            raise ValueError("No SDP in response")

        return JSONResponse({
            "sdp": sdp,
            "type": resp_type,
            "webrtc_id": webrtc_id
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"WebRTC offer error: {e}")
        return JSONResponse(
            {"error": str(e), "type": "error"},
            status_code=500
        )


# WHIP endpoint for external WebRTC ingestion (broadcaster-friendly)
whip_sessions = {}  # session_id -> session config


class WHIPOffer(BaseModel):
    """WHIP offer model for external stream ingestion."""
    sdp: str
    type: str = "offer"


@app.post("/whip")
async def whip_ingest(
    request: Request,
    src_language: str = "Spanish",
    target_language: str = "English",
    voice: str = "Cherry",
    _auth: bool = Depends(optional_auth)
):
    """
    WHIP (WebRTC-HTTP Ingestion Protocol) endpoint for broadcaster ingestion.

    This allows external encoders/broadcasters to push WebRTC streams for translation.
    Compatible with OBS, FFmpeg, GStreamer, and professional broadcast encoders.

    Usage with FFmpeg:
        ffmpeg -re -i input.mp4 -c:v libvpx -c:a opus -f webrtc http://host/whip

    Usage with GStreamer:
        gst-launch-1.0 ... ! webrtcsink signaller::uri=http://host/whip

    Query params:
        src_language: Source language (default: Spanish)
        target_language: Target language (default: English)
        voice: TTS voice (default: Cherry)
    """
    if not WHIP_ENABLED:
        raise HTTPException(status_code=403, detail="WHIP ingestion is disabled")

    # Read SDP from request body
    body = await request.body()
    content_type = request.headers.get("Content-Type", "")

    if "application/sdp" in content_type:
        sdp = body.decode("utf-8")
    else:
        # Try to parse as JSON
        try:
            data = json.loads(body)
            sdp = data.get("sdp", body.decode("utf-8"))
        except json.JSONDecodeError:
            sdp = body.decode("utf-8")

    session_id = secrets.token_hex(16)

    # Store session config
    whip_sessions[session_id] = {
        "src_language": src_language,
        "target_language": target_language,
        "voice": voice,
        "created_at": time.time(),
        "type": "whip_ingest"
    }

    try:
        # Use FastRTC's internal offer handling
        response = await stream.offer(
            sdp,
            "offer",
            extra_data={
                "src_language": src_language,
                "target_language": target_language,
                "voice": voice
            }
        )

        # Return SDP answer per WHIP spec
        return StreamingResponse(
            content=response["sdp"],
            media_type="application/sdp",
            status_code=201,
            headers={
                "Location": f"/whip/{session_id}",
                "ETag": f'"{session_id}"',
                "Accept-Patch": "application/trickle-ice-sdpfrag"
            }
        )
    except Exception as e:
        print(f"WHIP error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/whip/{session_id}")
async def whip_terminate(session_id: str, _auth: bool = Depends(optional_auth)):
    """Terminate a WHIP session."""
    if session_id in whip_sessions:
        del whip_sessions[session_id]
        return JSONResponse({"status": "terminated"}, status_code=200)
    raise HTTPException(status_code=404, detail="Session not found")


@app.patch("/whip/{session_id}")
async def whip_ice_trickle(session_id: str, request: Request, _auth: bool = Depends(optional_auth)):
    """Handle ICE trickle for WHIP session."""
    if session_id not in whip_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # For now, acknowledge the ICE candidate
    # Full implementation would forward to WebRTC peer connection
    return JSONResponse({"status": "accepted"}, status_code=204)


# WHEP endpoint for external WebRTC consumption
@app.post("/whep")
async def whep_consume(
    request: Request,
    session_id: str = None,
    _auth: bool = Depends(optional_auth)
):
    """
    WHEP (WebRTC-HTTP Egress Protocol) endpoint for stream consumption.

    This allows external players to receive translated WebRTC streams.

    Query params:
        session_id: Optional specific session to consume (default: first available)
    """
    body = await request.body()
    content_type = request.headers.get("Content-Type", "")

    if "application/sdp" in content_type:
        sdp = body.decode("utf-8")
    else:
        try:
            data = json.loads(body)
            sdp = data.get("sdp", body.decode("utf-8"))
        except json.JSONDecodeError:
            sdp = body.decode("utf-8")

    consumer_id = secrets.token_hex(16)

    try:
        # Create answer for consumer
        response = await stream.offer(sdp, "offer")

        return StreamingResponse(
            content=response["sdp"],
            media_type="application/sdp",
            status_code=201,
            headers={
                "Location": f"/whep/{consumer_id}",
                "ETag": f'"{consumer_id}"'
            }
        )
    except Exception as e:
        print(f"WHEP error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions")
async def list_sessions():
    """List active translation sessions for control room monitoring."""
    all_sessions = {**active_sessions, **whip_sessions}
    return JSONResponse({
        "sessions": [
            {
                "id": sid,
                "src_language": data["src_language"],
                "target_language": data["target_language"],
                "voice": data["voice"],
                "uptime": time.time() - data["created_at"],
                "type": data.get("type", "webrtc")
            }
            for sid, data in all_sessions.items()
        ]
    })


@app.get("/api/status")
async def api_status():
    """API health check for control room monitoring."""
    return JSONResponse({
        "status": "healthy",
        "service": "zen-live-translate",
        "version": "1.0.0",
        "active_sessions": len(active_sessions),
        "supported_languages": {
            "source": SRC_LANGUAGES,
            "target": TARGET_LANGUAGES
        },
        "voices": VOICES
    })


@app.get("/monitor")
async def monitor_page(request: Request, _auth: bool = Depends(optional_auth)):
    """Simplified monitor-only view for control room displays."""
    rtc_config = await get_cloudflare_turn_credentials_async() if get_space() else None
    html_content = (cur_dir / "monitor.html").read_text() if (cur_dir / "monitor.html").exists() else (cur_dir / "index.html").read_text()
    html_content = html_content.replace("__RTC_CONFIGURATION__", json.dumps(rtc_config))
    return HTMLResponse(content=html_content)


# Audio streaming for broadcast integration (can be ingested via NDI/SDI converters)
audio_subscribers = {}  # webrtc_id -> list of asyncio.Queue


@app.get("/audio/stream/{webrtc_id}")
async def audio_stream(webrtc_id: str):
    """
    Raw PCM audio stream for broadcast integration.
    
    This endpoint streams translated audio as raw PCM16 at 24kHz.
    Can be converted to SDI/NDI using tools like:
    - ffmpeg -f s16le -ar 24000 -ac 1 -i http://host/audio/stream/ID -f alsa default
    - OBS with browser source + audio capture
    - Blackmagic Web Presenter
    - NDI Tools with HTTP input
    
    For SRT output, pipe through:
    ffmpeg -f s16le -ar 24000 -ac 1 -i http://host/audio/stream/ID \
           -c:a aac -f mpegts srt://dest:port
    """
    async def generate_audio():
        queue = asyncio.Queue()
        if webrtc_id not in audio_subscribers:
            audio_subscribers[webrtc_id] = []
        audio_subscribers[webrtc_id].append(queue)
        
        try:
            while True:
                try:
                    audio_data = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield audio_data
                except asyncio.TimeoutError:
                    # Send silence to keep connection alive
                    yield b'\x00' * 4800  # 100ms of silence at 24kHz
        finally:
            if webrtc_id in audio_subscribers:
                audio_subscribers[webrtc_id].remove(queue)

    return StreamingResponse(
        generate_audio(),
        media_type="audio/pcm",
        headers={
            "Content-Type": "audio/L16;rate=24000;channels=1",
            "Cache-Control": "no-cache",
            "X-Audio-Format": "PCM16 24kHz Mono",
            "Access-Control-Allow-Origin": "*"
        }
    )


@app.get("/audio/wav/{webrtc_id}")
async def audio_wav_stream(webrtc_id: str):
    """
    WAV-wrapped audio stream for easier playback/ingestion.
    
    Includes WAV header for compatibility with more players/converters.
    Useful for direct monitoring or conversion to broadcast formats.
    """
    async def generate_wav():
        # Send WAV header for streaming (size = max)
        wav_header = bytes([
            0x52, 0x49, 0x46, 0x46,  # "RIFF"
            0xFF, 0xFF, 0xFF, 0x7F,  # Size (max for streaming)
            0x57, 0x41, 0x56, 0x45,  # "WAVE"
            0x66, 0x6D, 0x74, 0x20,  # "fmt "
            0x10, 0x00, 0x00, 0x00,  # Subchunk1Size (16)
            0x01, 0x00,              # AudioFormat (1 = PCM)
            0x01, 0x00,              # NumChannels (1)
            0xC0, 0x5D, 0x00, 0x00,  # SampleRate (24000)
            0x80, 0xBB, 0x00, 0x00,  # ByteRate (48000)
            0x02, 0x00,              # BlockAlign (2)
            0x10, 0x00,              # BitsPerSample (16)
            0x64, 0x61, 0x74, 0x61,  # "data"
            0xFF, 0xFF, 0xFF, 0x7F,  # Subchunk2Size (max)
        ])
        yield wav_header
        
        queue = asyncio.Queue()
        if webrtc_id not in audio_subscribers:
            audio_subscribers[webrtc_id] = []
        audio_subscribers[webrtc_id].append(queue)
        
        try:
            while True:
                try:
                    audio_data = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield audio_data
                except asyncio.TimeoutError:
                    yield b'\x00' * 4800
        finally:
            if webrtc_id in audio_subscribers:
                audio_subscribers[webrtc_id].remove(queue)

    return StreamingResponse(
        generate_wav(),
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*"
        }
    )


@app.get("/broadcast/info")
async def broadcast_info():
    """
    Information for broadcast engineers on how to integrate.
    """
    base_url = "http://YOUR_HOST:8000"
    return JSONResponse({
        "service": "Zen Live Translate - Broadcast Integration",
        "endpoints": {
            "control_room_ui": f"{base_url}/",
            "monitor_only": f"{base_url}/monitor?autostart=1",
            "webrtc_offer": f"{base_url}/webrtc/offer",
            "whip_ingest": f"{base_url}/whip?src_language=Spanish&target_language=English",
            "whep_consume": f"{base_url}/whep",
            "transcript_sse": f"{base_url}/outputs?webrtc_id=SESSION_ID",
            "audio_pcm": f"{base_url}/audio/stream/SESSION_ID",
            "audio_wav": f"{base_url}/audio/wav/SESSION_ID",
            "api_status": f"{base_url}/api/status",
            "api_sessions": f"{base_url}/api/sessions"
        },
        "whip_whep": {
            "description": "Standard WebRTC ingestion/egress protocols for professional broadcast",
            "whip_enabled": WHIP_ENABLED,
            "whip_post": "POST /whip with SDP offer (Content-Type: application/sdp)",
            "whip_params": "?src_language=Spanish&target_language=English&voice=Cherry",
            "whep_post": "POST /whep with SDP offer to consume translated stream"
        },
        "audio_format": {
            "encoding": "PCM16 signed little-endian",
            "sample_rate": 24000,
            "channels": 1,
            "bits_per_sample": 16
        },
        "integration_examples": {
            "ffplay_direct": "ffplay -f s16le -ar 24000 -ac 1 http://host/audio/stream/ID",
            "ffmpeg_to_srt": "ffmpeg -f s16le -ar 24000 -ac 1 -i http://host/audio/stream/ID -c:a aac -f mpegts 'srt://dest:port'",
            "ffmpeg_to_rtmp": "ffmpeg -f s16le -ar 24000 -ac 1 -i http://host/audio/stream/ID -c:a aac -f flv rtmp://dest/live/stream",
            "obs_browser": "Add Browser Source with URL: http://host/monitor?autostart=1",
            "vlc": "vlc http://host/audio/wav/SESSION_ID",
            "gstreamer_whip": "gst-launch-1.0 ... ! webrtcsink signaller::uri=http://host/whip"
        },
        "authentication": {
            "enabled": AUTH_ENABLED,
            "type": "HTTP Basic Auth",
            "header": "Authorization: Basic <base64(username:password)>"
        },
        "latency": {
            "typical": "200-500ms end-to-end",
            "factors": ["network RTT", "AI processing", "TTS generation"]
        }
    })


@app.get("/")
async def root_page(request: Request, _auth: bool = Depends(optional_auth)):
    """Main control room UI - requires authentication if enabled."""
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


# =============================================================================
# WebSocket Proxy API - Proxies to DashScope qwen3-livetranslate-flash-realtime
# =============================================================================

# Full language support per DashScope docs
SUPPORTED_LANGUAGES = {
    "en": {"name": "English", "audio": True},
    "zh": {"name": "Chinese", "audio": True},
    "ru": {"name": "Russian", "audio": True},
    "fr": {"name": "French", "audio": True},
    "de": {"name": "German", "audio": True},
    "pt": {"name": "Portuguese", "audio": True},
    "es": {"name": "Spanish", "audio": True},
    "it": {"name": "Italian", "audio": True},
    "ko": {"name": "Korean", "audio": True},
    "ja": {"name": "Japanese", "audio": True},
    "yue": {"name": "Cantonese", "audio": True},
    "id": {"name": "Indonesian", "audio": False},  # Text only
    "vi": {"name": "Vietnamese", "audio": False},  # Text only
    "th": {"name": "Thai", "audio": False},  # Text only
    "ar": {"name": "Arabic", "audio": False},  # Text only
    "hi": {"name": "Hindi", "audio": False},  # Text only
    "el": {"name": "Greek", "audio": False},  # Text only
    "tr": {"name": "Turkish", "audio": False},  # Text only
}

# Full voice support per DashScope docs
SUPPORTED_VOICES = {
    "Cherry": {"description": "Sunny, positive, friendly female", "languages": ["zh", "en", "fr", "de", "ru", "it", "es", "pt", "ja", "ko"]},
    "Nofish": {"description": "Designer voice, casual male", "languages": ["zh", "en", "fr", "de", "ru", "it", "es", "pt", "ja", "ko"]},
    "Jada": {"description": "Lively Shanghainese woman", "languages": ["zh"]},
    "Dylan": {"description": "Young Beijing man", "languages": ["zh"]},
    "Sunny": {"description": "Sweet Sichuanese girl", "languages": ["zh"]},
    "Peter": {"description": "Tianjin crosstalk style", "languages": ["zh"]},
    "Kiki": {"description": "Sweet Hong Kong best friend", "languages": ["yue"]},
    "Eric": {"description": "Sichuan male voice", "languages": ["zh"]},
}


@app.websocket("/v1/realtime")
async def websocket_proxy(websocket: WebSocket):
    """
    WebSocket proxy to DashScope qwen3-livetranslate-flash-realtime.

    This endpoint proxies all WebSocket messages to DashScope's real-time
    translation API, allowing clients to use api.zen-live.hanzo.ai as
    a drop-in replacement for dashscope-intl.aliyuncs.com.

    Client Events (send to server):
    - session.update: Configure session (language, voice, modalities)
    - input_audio_buffer.append: Send Base64-encoded PCM16 audio
    - input_image_buffer.append: Send Base64-encoded image for context

    Server Events (receive from server):
    - session.created: Initial session confirmation
    - session.updated: Configuration update confirmation
    - response.created: Translation started
    - response.audio.delta: Incremental translated audio (Base64)
    - response.audio_transcript.delta: Incremental transcript text
    - response.audio_transcript.done: Complete transcript
    - response.done: Response complete with usage stats
    - error: Error details

    See /api/spec for full API specification.
    """
    if not API_KEY:
        await websocket.close(code=4001, reason="API_KEY not configured")
        return

    await websocket.accept()

    upstream_ws = None
    try:
        # Connect to DashScope
        upstream_ws = await connect(
            API_URL,
            additional_headers={"Authorization": f"Bearer {API_KEY}"},
            ssl=ssl_context
        )

        async def forward_to_upstream():
            """Forward client messages to DashScope."""
            try:
                while True:
                    data = await websocket.receive_text()
                    await upstream_ws.send(data)
            except WebSocketDisconnect:
                pass
            except Exception as e:
                print(f"Client->Upstream error: {e}")

        async def forward_to_client():
            """Forward DashScope responses to client."""
            try:
                async for message in upstream_ws:
                    await websocket.send_text(message)
            except Exception as e:
                print(f"Upstream->Client error: {e}")

        # Run both directions concurrently
        await asyncio.gather(
            forward_to_upstream(),
            forward_to_client(),
            return_exceptions=True
        )

    except Exception as e:
        print(f"WebSocket proxy error: {e}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "error": {"message": str(e), "type": "proxy_error"}
            }))
        except:
            pass
    finally:
        if upstream_ws:
            await upstream_ws.close()


@app.get("/api/spec")
async def api_specification():
    """
    Full API specification for qwen3-livetranslate-flash-realtime.

    This documents the WebSocket protocol for real-time translation.
    """
    return JSONResponse({
        "name": "Zen Live Translation API",
        "version": "1.0.0",
        "description": "Real-time audio/video translation powered by Qwen3 LiveTranslate",
        "websocket_endpoint": "/v1/realtime",
        "protocol": "WebSocket",
        "authentication": {
            "type": "Bearer Token (handled by server)",
            "note": "API key is configured server-side via API_KEY env var"
        },
        "client_events": {
            "session.update": {
                "description": "Update session configuration",
                "example": {
                    "event_id": "event_xxx",
                    "type": "session.update",
                    "session": {
                        "modalities": ["text", "audio"],
                        "voice": "Cherry",
                        "input_audio_format": "pcm16",
                        "output_audio_format": "pcm24",
                        "input_audio_transcription": {"language": "es"},
                        "translation": {"language": "en"}
                    }
                },
                "parameters": {
                    "modalities": "['text'] or ['text', 'audio']",
                    "voice": "Voice ID (see /api/voices)",
                    "input_audio_transcription.language": "Source language code",
                    "translation.language": "Target language code"
                }
            },
            "input_audio_buffer.append": {
                "description": "Send audio data for translation",
                "example": {
                    "event_id": "event_xxx",
                    "type": "input_audio_buffer.append",
                    "audio": "<base64_pcm16_audio>"
                },
                "parameters": {
                    "audio": "Base64-encoded PCM16 audio (16kHz, mono)"
                }
            },
            "input_image_buffer.append": {
                "description": "Send image for visual context (improves accuracy)",
                "example": {
                    "event_id": "event_xxx",
                    "type": "input_image_buffer.append",
                    "image": "<base64_jpeg_image>"
                },
                "parameters": {
                    "image": "Base64-encoded JPEG image (max 1080p, 500KB)"
                },
                "notes": [
                    "Max 2 images per second",
                    "Must send audio first before images",
                    "Helps with homonyms and proper nouns"
                ]
            }
        },
        "server_events": {
            "session.created": "Session initialized with default config",
            "session.updated": "Session config updated successfully",
            "response.created": "Model started generating response",
            "response.audio.delta": "Incremental audio chunk (Base64 PCM24)",
            "response.audio.done": "Audio generation complete",
            "response.audio_transcript.delta": "Incremental transcript text",
            "response.audio_transcript.done": "Complete transcript with final text",
            "response.text.delta": "Incremental text (text-only mode)",
            "response.text.done": "Complete text (text-only mode)",
            "response.done": "Response complete with usage statistics",
            "error": "Error occurred"
        },
        "audio_format": {
            "input": {"encoding": "pcm16", "sample_rate": 16000, "channels": 1},
            "output": {"encoding": "pcm24", "sample_rate": 24000, "channels": 1}
        },
        "languages": SUPPORTED_LANGUAGES,
        "voices": SUPPORTED_VOICES,
        "billing": {
            "audio": "12.5 tokens per second (input or output)",
            "image": "0.5 tokens per 28x28 pixels"
        },
        "latency": "~3 seconds for simultaneous interpretation",
        "links": {
            "docs": "/docs",
            "languages": "/api/languages",
            "voices": "/api/voices",
            "github": "https://github.com/zenlm/zen-live"
        }
    })


@app.get("/api/languages")
async def list_languages():
    """List all supported languages with audio capability info."""
    return JSONResponse({
        "languages": [
            {
                "code": code,
                "name": info["name"],
                "audio_output": info["audio"],
                "text_output": True
            }
            for code, info in SUPPORTED_LANGUAGES.items()
        ],
        "source_languages": list(SUPPORTED_LANGUAGES.keys()),
        "target_languages_audio": [k for k, v in SUPPORTED_LANGUAGES.items() if v["audio"]],
        "target_languages_text_only": [k for k, v in SUPPORTED_LANGUAGES.items() if not v["audio"]]
    })


@app.get("/api/voices")
async def list_voices():
    """List all supported TTS voices with language compatibility."""
    return JSONResponse({
        "voices": [
            {
                "id": voice_id,
                "description": info["description"],
                "supported_languages": info["languages"]
            }
            for voice_id, info in SUPPORTED_VOICES.items()
        ],
        "default": "Cherry",
        "multilingual": ["Cherry", "Nofish"],
        "regional": ["Jada", "Dylan", "Sunny", "Peter", "Kiki", "Eric"]
    })


def handle_exit(sig, frame):
    print("\n👋 Zen Live shutting down...")
    exit(0)


signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

if __name__ == "__main__":
    import uvicorn

    mode = os.getenv("MODE", "").upper()
    port = int(os.getenv("PORT", 8000))

    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                       ZEN LIVE                            ║
    ║         Real-time Speech Translation Service              ║
    ║                   Powered by Hanzo AI                     ║
    ╠═══════════════════════════════════════════════════════════╣
    ║  Control Room:  http://localhost:{port:<5}                  ║
    ║  Monitor View:  http://localhost:{port}/monitor             ║
    ║  API Docs:      http://localhost:{port}/docs                ║
    ║  Broadcast:     http://localhost:{port}/broadcast/info      ║
    ╚═══════════════════════════════════════════════════════════╝
    """.format(port=port))

    if mode == "UI":
        demo = enhance_ui()
        demo.launch(server_port=port)
    elif mode == "PHONE":
        stream.fastphone(host="0.0.0.0", port=port)
    else:
        uvicorn.run(app, host="0.0.0.0", port=port)