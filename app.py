"""
Zen Live - Real-time Speech Translation for Broadcast

A low-latency simultaneous translation service for news control rooms.
Powered by Hanzo AI infrastructure with Zen Omni backend.

Backend options:
  1. Hanzo Node API (recommended) - Set HANZO_NODE_URL
  2. Direct Hanzo API - Set HANZO_API_KEY
  3. Local Zen Omni model - Set ZEN_OMNI_PATH

Usage:
  export HANZO_NODE_URL=http://localhost:3690  # or
  export HANZO_API_KEY=your_hanzo_key
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
from contextlib import asynccontextmanager

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
from fastrtc.stream import Body as RTCBody
from gradio.utils import get_space
from websockets.asyncio.client import connect
import ssl
import certifi

import cv2
import subprocess
import shutil

load_dotenv()

os.environ["MODE"] = os.environ.get("MODE", "")
cur_dir = Path(__file__).parent

# Backend configuration
HANZO_NODE_URL = os.environ.get("HANZO_NODE_URL")  # Preferred: Hanzo Node backend
HANZO_API_KEY = os.environ.get("HANZO_API_KEY")  # Fallback: Direct Hanzo API
ZEN_OMNI_PATH = os.environ.get("ZEN_OMNI_PATH")  # Optional: Local model path

# Server configuration
BASE_URL = os.environ.get("BASE_URL", "https://zen-live.hanzo.ai")  # Public URL for docs
PORT = int(os.environ.get("PORT", 8000))

# Authentication (optional - set both to enable)
AUTH_USER = os.environ.get("ZEN_LIVE_USER")  # e.g., admin
AUTH_PASS = os.environ.get("ZEN_LIVE_PASS")  # e.g., livedemo2025
AUTH_ENABLED = bool(AUTH_USER and AUTH_PASS)

# Input source configuration (defaults, can be overridden via web UI localStorage)
SRT_INPUT_URL = os.environ.get("SRT_INPUT_URL")  # e.g., srt://source:9000?mode=caller
RTMP_INPUT_URL = os.environ.get("RTMP_INPUT_URL")  # e.g., rtmp://source/live/stream
WHIP_ENABLED = os.environ.get("WHIP_ENABLED", "true").lower() == "true"  # WebRTC WHIP input

TRANSLATE_API_URL = os.environ.get("TRANSLATE_API_URL", "")
ASR_API_URL = os.environ.get("ASR_API_URL", "")

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


async def verify_websocket_auth(websocket: WebSocket) -> bool:
    """Verify WebSocket authentication via query param or header."""
    if not AUTH_ENABLED:
        return True

    # Check for auth in query params (for WebSocket clients that can't set headers)
    token = websocket.query_params.get("token")
    if token:
        try:
            credentials = base64.b64decode(token).decode("utf-8")
            username, password = credentials.split(":", 1)
            if secrets.compare_digest(username, AUTH_USER) and secrets.compare_digest(password, AUTH_PASS):
                return True
        except Exception:
            pass

    # Check for Authorization header
    auth_header = websocket.headers.get("authorization")
    if auth_header and auth_header.startswith("Basic "):
        try:
            credentials = base64.b64decode(auth_header[6:]).decode("utf-8")
            username, password = credentials.split(":", 1)
            if secrets.compare_digest(username, AUTH_USER) and secrets.compare_digest(password, AUTH_PASS):
                return True
        except Exception:
            pass

    return False


VOICES = ["Cherry", "Nofish", "Jada", "Dylan", "Sunny", "Peter", "Kiki", "Eric"]

ssl_context = ssl.create_default_context(cafile=certifi.where())

# Determine backend and get credentials
if HANZO_NODE_URL:
    print(f"🔗 Zen Live: Using Hanzo Node backend at {HANZO_NODE_URL}")
    BACKEND_TYPE = "hanzo_node"
elif HANZO_API_KEY:
    print("🔗 Zen Live: Using direct Hanzo API")
    BACKEND_TYPE = "hanzo_api"
else:
    print("⚠️  Zen Live: No backend configured")
    print("   Set HANZO_NODE_URL (recommended) or HANZO_API_KEY")
    BACKEND_TYPE = None

headers = {"Authorization": f"Bearer {HANZO_API_KEY}"} if HANZO_API_KEY else {}
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

# Handler registry to track session_id -> handler mapping for concurrent users
handler_registry = {}  # session_id -> LiveTranslateHandler
# Pending session ID to be assigned to the next handler that starts
pending_session_id = None
# SRC_LANGUAGES = ["en", "zh", "ru", "fr", "de", "pt", "es", "it", "ko", "ja", "yue", "id", "vi", "th", "ar", "hi", "el", "tr"]
# TARGET_LANGUAGES = ["en", "zh", "ru", "fr", "de", "pt", "es", "it", "ko", "ja", "yue", "id", "vi", "th", "ar"]

# Spanish first for news monitoring default
SRC_LANGUAGES = [LANG_MAP[code] for code in ["es", "en", "zh", "pt", "fr", "de", "ru", "it", "ko", "ja", "yue", "id", "vi", "th", "ar", "hi", "el", "tr"]]
TARGET_LANGUAGES = [LANG_MAP[code] for code in ["en", "zh", "ru", "fr", "de", "pt", "es", "it", "ko", "ja", "yue", "id", "vi", "th", "ar"]]


class LiveTranslateHandler(AsyncAudioVideoStreamHandler):
    def __init__(self, session_config: dict = None, session_id: str = None) -> None:
        print("🔧 LiveTranslateHandler.__init__() called")
        super().__init__(
            expected_layout="mono",
            output_sample_rate=24_000,
            input_sample_rate=16_000,
        )
        self.connection = None
        self.output_queue = asyncio.Queue()
        self.video_queue = asyncio.Queue()

        self.last_send_time = 0.0     # Last send timestamp
        self.video_interval = 0.5     # Interval in seconds (match reference: 2fps for context)
        self.latest_frame = None

        self.awaiting_new_message = True
        self.stable_text = ""  # Confirmed text (black)
        self.temp_text = ""    # Interim text (gray)
        # Source language transcript (English input)
        self.source_text = ""
        self.source_temp = ""

        # Session ID for tracking and audio routing
        self.session_id = session_id

        # Session-specific config (avoids race conditions with concurrent users)
        self.session_config = session_config or {
            "src_language": "Spanish",
            "target_language": "English",
            "voice": "Nofish"
        }
        print(f"🔧 LiveTranslateHandler.__init__() complete, session_id: {session_id}, config: {self.session_config}")


    def copy(self):
        global pending_session_id
        print(f"🔧 LiveTranslateHandler.copy() called - creating new handler instance, pending_session_id: {pending_session_id}")
        # Pass session config and pending session_id to the new handler instance
        new_handler = LiveTranslateHandler(
            session_config=self.session_config.copy() if self.session_config else None,
            session_id=pending_session_id
        )
        # Register handler if we have a session_id
        if pending_session_id:
            handler_registry[pending_session_id] = new_handler
            pending_session_id = None  # Clear pending
        return new_handler

    _msg_counter = 0

    @classmethod
    def msg_id(cls) -> str:
        cls._msg_counter += 1
        return f"evt_{cls._msg_counter}"

    async def start_up(self):
        try:
            # Use global current_session_config (set by webrtc_offer endpoint)
            # This is set just before stream.offer() is called
            config = current_session_config
            print(f"🎤 Handler starting with session_config: {config}")

            src_language_name = config.get("src_language", "Spanish")
            target_language_name = config.get("target_language", "English")
            voice_id = config.get("voice", "Nofish")

            src_language_code = LANG_MAP_REVERSE.get(src_language_name, "en")
            target_language_code = LANG_MAP_REVERSE.get(target_language_name, "es")

            print(f"🌐 Translation: {src_language_name} ({src_language_code}) -> {target_language_name} ({target_language_code}), Voice: {voice_id}")

            if src_language_code == target_language_code:
                print(f"⚠️ Source and target language are the same ({target_language_name}), running in echo mode")

            print(f"🔌 Connecting to Hanzo WebSocket: {TRANSLATE_API_URL[:50]}...")
            async with connect(TRANSLATE_API_URL, additional_headers=headers, ssl=ssl_context) as conn:
                print("✅ Hanzo WebSocket connected!")
                self.client = conn
                session_update_msg = {
                    "event_id": self.msg_id(),
                    "type": "session.update",
                    "session": {
                        "modalities": ["text", "audio"],
                        "voice": voice_id,
                        "input_audio_format": "pcm16",
                        "output_audio_format": "pcm16",
                        "input_audio_transcription": {
                            "language": src_language_code
                        },
                        "translation": {
                            "source_language": src_language_code,
                            "language": target_language_code
                        }
                    },
                }
                print(f"📤 Sending session.update: {json.dumps(session_update_msg, indent=2)}")
                await conn.send(json.dumps(session_update_msg))
                self.connection = conn
                print("🎧 Now listening for Hanzo responses...")

                # Each WebSocket response (data) is a JSON event representing translation progress
                async for data in self.connection:
                    print(f"📨 Hanzo response: {data[:200]}..." if len(data) > 200 else f"📨 Hanzo response: {data}")
                    event = json.loads(data)
                    if "type" not in event:
                        continue
                    event_type = event["type"]

                    # Handle source language transcription (input)
                    if event_type in ("input_audio_transcript.delta", "input_audio_transcript.text"):
                        self.source_text = event.get("text", "") or ""
                        self.source_temp = event.get("stash", "") or ""
                        print(f"[SOURCE] {self.source_text} {self.source_temp}")
                        # Send source transcript to client
                        await self.output_queue.put(
                            AdditionalOutputs({
                                "role": "source",
                                "source_content": f"{self.source_text}{self.source_temp}",
                                "update": True
                            })
                        )

                    elif event_type == "input_audio_transcript.done":
                        transcript = event.get("transcript", "")
                        print(f"[SOURCE DONE] {transcript}")
                        if transcript:
                            self.source_text = transcript
                            self.source_temp = ""
                            await self.output_queue.put(
                                AdditionalOutputs({
                                    "role": "source",
                                    "source_content": transcript,
                                    "update": True
                                })
                            )

                    elif event_type in ("response.text.text", "response.audio_transcript.text"):
                        # Update stable text (stash/text are confirmed)
                        self.stable_text = event.get("text", "") or ""
                        self.temp_text = event.get("stash", "") or ""
                        # self.stable_text = event.get("stash", "") or ""
                        # self.temp_text = event.get("text", "") or ""

                        print(f"[STABLE] {self.stable_text}")
                        print(f"[TEMP] {self.temp_text}")
                        await self.output_queue.put(
                            AdditionalOutputs({
                                "role": "assistant",
                                # Stable text is black, interim text is gray
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
                        # Start new message bubble
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
                            # Publish to HTTP audio stream subscribers for broadcast output
                            # Uses session_id for proper routing to concurrent sessions
                            await publish_audio_to_subscribers(audio_data, self.session_id)


        except Exception as e:
            print(f"Connection error: {e}")
            await self.shutdown()

    # Client -> Server: receive video frame
    async def video_receive(self, frame: np.ndarray):
        self.latest_frame = frame
        if self.connection is None:
            return

        # Push frame to local queue for display immediately (non-blocking)
        try:
            self.video_queue.put_nowait(frame)
        except asyncio.QueueFull:
            pass  # Drop frame if queue full

        now = time.time()
        if now - self.last_send_time < self.video_interval:
            return
        self.last_send_time = now

        # Process video in background task to not block audio
        asyncio.create_task(self._send_video_frame(frame))

    async def _send_video_frame(self, frame: np.ndarray):
        """Send video frame to API - runs in background to not block audio."""
        try:
            if self.connection is None:
                return
            # Resize and encode (CPU-bound work)
            frame_resized = cv2.resize(frame, (640, 360))
            _, buf = cv2.imencode(".jpg", frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            jpeg_bytes = buf.tobytes()
            img_b64 = base64.b64encode(jpeg_bytes).decode()

            # Publish to video stream subscribers for remote viewers
            if self.session_id:
                await publish_video_to_subscribers(jpeg_bytes, self.session_id)

            # Send to API
            msg = f'{{"event_id":"evt_{self._msg_counter}","type":"input_image_buffer.append","image":"{img_b64}"}}'
            self._msg_counter += 1
            await self.connection.send(msg)
        except Exception as e:
            pass  # Don't let video errors affect audio

    # Server -> Client: emit video frame
    async def video_emit(self):
        if self.latest_frame is not None:
            return self.latest_frame
        return np.zeros((100, 100, 3), dtype=np.uint8)


    async def receive(self, frame):
        # Audio receive - send immediately with minimal overhead
        if self.connection is None:
            return
        sr, array = frame
        audio_b64 = base64.b64encode(array.squeeze().tobytes()).decode()
        # Use string formatting instead of json.dumps for speed
        msg = f'{{"event_id":"evt_{self._msg_counter}","type":"input_audio_buffer.append","audio":"{audio_b64}"}}'
        self._msg_counter += 1
        await self.connection.send(msg)


    async def emit(self) -> tuple[int, np.ndarray] | AdditionalOutputs | None:
        return await wait_for_item(self.output_queue)

    async def shutdown(self) -> None:
        """Close connection and cleanup resources"""

        if self.connection:
            await self.connection.close()
            self.connection = None

        # Clear queue
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



chatbot = gr.Chatbot(type="messages", allow_tags=False)
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

# Optional: Temporarily disable TURN configuration for testing
rtc_config = get_cloudflare_turn_credentials_async if get_space() else None
# rtc_config = None  # Uncomment to disable TURN for testing


stream = Stream(
    LiveTranslateHandler(),
    mode="send-receive",
    modality="audio-video",
    additional_inputs=[src_language, language, voice, chatbot],
    additional_outputs=[chatbot],
    additional_outputs_handler=update_chatbot,
    rtc_configuration=rtc_config,
    # High limits - FastRTC treats None as 1, so set explicit high values
    concurrency_limit=1000,
    time_limit=86400,  # 24 hours
)

# Frontend UI customization
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

# Generate dynamic documentation based on environment
auth_header = f"Authorization: Basic {base64.b64encode(f'{AUTH_USER}:{AUTH_PASS}'.encode()).decode()}" if AUTH_ENABLED else "No authentication required"
ws_url = BASE_URL.replace("https", "wss").replace("http", "ws")

# Build documentation string separately to avoid f-string issues
auth_section = ""
if AUTH_ENABLED:
    auth_section = f"""HTTP Basic Auth is required for all protected endpoints.

**Using the Authorization Header:**
```
{auth_header}
```

**Examples:**
```bash
# API Status
curl -H '{auth_header}' \\
     {BASE_URL}/api/status

# WebSocket connection
wscat -c '{ws_url}/v1/realtime' \\
      -H '{auth_header}'

# Stream audio output (replace SESSION_ID with actual session)
curl -H '{auth_header}' \\
     {BASE_URL}/audio/stream/SESSION_ID
```"""
else:
    auth_section = """No authentication required - open access

## API Examples:
```bash
# API Status
curl {BASE_URL}/api/status

# WebSocket connection
wscat -c '{ws_url}/v1/realtime'

# Stream audio output (replace SESSION_ID with actual session)
curl {BASE_URL}/audio/stream/SESSION_ID
```""".format(BASE_URL=BASE_URL, ws_url=ws_url)

app_description = f"""
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
{auth_section}

## Session Management
When you connect via WebRTC or WHIP, you receive a `webrtc_id` (session ID) in the response.
Use this SESSION_ID to access:
- Transcripts: `{BASE_URL}/outputs?webrtc_id=SESSION_ID`
- Audio stream: `{BASE_URL}/audio/stream/SESSION_ID`
- WAV audio: `{BASE_URL}/audio/wav/SESSION_ID`

## Broadcast Integration
See [/broadcast/info](/broadcast/info) for ffmpeg, SRT, RTMP, and NDI integration examples.

Powered by [Hanzo AI](https://hanzo.ai) | [GitHub](https://github.com/zenlm/zen-live)
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    asyncio.create_task(cleanup_stale_sessions())
    print("🚀 Started session cleanup background task")
    yield
    # Shutdown (cleanup if needed)
    print("👋 Zen Live shutting down...")


app = FastAPI(
    title="Zen Live",
    description=app_description,
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
    },
    lifespan=lifespan
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
    voice: str = "Nofish"


# Store active sessions for control room
active_sessions = {}
# Session configs indexed by session_id - used by handlers to get their config
session_configs = {}
# Current session config for single-instance mode (fallback)
current_session_config = {"src_language": "Spanish", "target_language": "English", "voice": "Nofish"}
# Session timeout in seconds (cleanup after 1 hour of inactivity)
SESSION_TIMEOUT = 3600


async def cleanup_stale_sessions():
    """Background task to cleanup stale sessions and prevent memory leak."""
    while True:
        await asyncio.sleep(300)  # Run every 5 minutes
        current_time = time.time()
        stale_sessions = []

        # Find stale active_sessions
        for session_id, session_data in list(active_sessions.items()):
            created_at = session_data.get("created_at", 0)
            if current_time - created_at > SESSION_TIMEOUT:
                stale_sessions.append(session_id)

        # Remove stale sessions
        for session_id in stale_sessions:
            active_sessions.pop(session_id, None)
            session_configs.pop(session_id, None)
            handler_registry.pop(session_id, None)  # Also cleanup handler registry
            audio_subscribers.pop(session_id, None)  # Cleanup audio subscribers
            print(f"🧹 Cleaned up stale session: {session_id[:8]}...")

        if stale_sessions:
            print(f"🧹 Cleaned up {len(stale_sessions)} stale session(s)")


@app.post("/api/webrtc/offer")
async def webrtc_offer(offer: WebRTCOffer):
    """Handle WebRTC offer from control room portal.

    This endpoint receives SDP offers and returns SDP answers
    for establishing low-latency WebRTC connections.
    """
    try:
        # Generate session ID
        session_id = secrets.token_hex(16)

        # Create session config for this specific user
        session_config = {
            "src_language": offer.src_language,
            "target_language": offer.target_language,
            "voice": offer.voice
        }

        # Store session config
        active_sessions[session_id] = {
            **session_config,
            "created_at": time.time()
        }

        # Store session config by ID so handler can look it up
        session_configs[session_id] = session_config.copy()

        # Also update global fallback for single-user mode
        global current_session_config, pending_session_id
        current_session_config = session_config.copy()
        # Set pending session_id so handler.copy() can pick it up
        pending_session_id = session_id

        print(f"📡 WebRTC offer received: {offer.src_language} -> {offer.target_language}, voice: {offer.voice}")

        # Use FastRTC's internal offer handling with proper Body object
        rtc_body = RTCBody(
            sdp=offer.sdp,
            type=offer.type,
            webrtc_id=session_id,
            candidate=None
        )
        print(f"📤 Calling stream.offer with session_id: {session_id}")
        response = await stream.offer(rtc_body)
        print(f"📥 stream.offer response type: {type(response)}")
        print(f"📥 stream.offer response: {response}")

        # Handle response - FastRTC returns dict on success, JSONResponse on error
        if isinstance(response, JSONResponse):
            # FastRTC returned an error response, pass it through
            print(f"📥 FastRTC returned JSONResponse (error case)")
            return response
        elif isinstance(response, dict):
            sdp = response.get("sdp", "")
            resp_type = response.get("type", "answer")
            webrtc_id = session_id
        else:
            # Unexpected response type - try to handle it
            print(f"⚠️ Unexpected response type: {type(response)}")
            if hasattr(response, 'body'):
                # It might be a Response object
                import json
                body = response.body.decode() if isinstance(response.body, bytes) else response.body
                try:
                    data = json.loads(body)
                    if 'error' in data or 'status' in data:
                        return response
                    sdp = data.get("sdp", "")
                    resp_type = data.get("type", "answer")
                except:
                    sdp = ""
                    resp_type = "answer"
            else:
                sdp = getattr(response, "sdp", "")
                resp_type = getattr(response, "type", "answer")
            webrtc_id = session_id

        print(f"📥 Extracted SDP (first 200 chars): {sdp[:200] if sdp else 'EMPTY'}")

        if not sdp:
            raise ValueError("No SDP in response - FastRTC may have returned an error")

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
        # Use FastRTC's internal offer handling with proper Body object
        whip_session_id = secrets.token_hex(16)
        rtc_body = RTCBody(
            sdp=sdp,
            type="offer",
            webrtc_id=whip_session_id,
            candidate=None
        )
        response = await stream.offer(rtc_body)

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
        # Create answer for consumer with proper Body object
        rtc_body = RTCBody(
            sdp=sdp,
            type="offer",
            webrtc_id=consumer_id,
            candidate=None
        )
        response = await stream.offer(rtc_body)

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
        "base_url": BASE_URL,
        "active_sessions": len(active_sessions),
        "authentication": {
            "enabled": AUTH_ENABLED,
            "type": "HTTP Basic Auth" if AUTH_ENABLED else "None",
            "header": "Authorization: Basic <base64(username:password)>" if AUTH_ENABLED else None,
            "curl_example": f"curl -u '<username>:<password>' {BASE_URL}/api/status" if AUTH_ENABLED else f"curl {BASE_URL}/api/status",
            "websocket_example": f"wscat -c '{BASE_URL.replace('https', 'wss').replace('http', 'ws')}/v1/realtime' -H 'Authorization: Basic <base64>'" if AUTH_ENABLED else f"wscat -c '{BASE_URL.replace('https', 'wss').replace('http', 'ws')}/v1/realtime'"
        },
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


@app.get("/view")
async def viewer_page(request: Request, session: str = None):
    """
    Remote viewer page - same as main page but in view-only mode.
    Share link: /view?session=SESSION_ID
    No authentication required for viewers.
    """
    rtc_config = await get_cloudflare_turn_credentials_async() if get_space() else None
    html_content = (cur_dir / "index.html").read_text()
    html_content = html_content.replace("__RTC_CONFIGURATION__", json.dumps(rtc_config))
    return HTMLResponse(content=html_content)


# Audio streaming for broadcast integration (can be ingested via NDI/SDI converters)
audio_subscribers = {}  # webrtc_id -> list of asyncio.Queue
# Global audio publisher for broadcast - publishes to all subscribers
all_audio_subscribers = []  # list of asyncio.Queue for /audio/broadcast

# Video streaming for remote viewers
video_subscribers = {}  # session_id -> list of asyncio.Queue


async def publish_video_to_subscribers(frame_jpeg: bytes, session_id: str):
    """Publish video frame to HTTP stream subscribers."""
    if session_id and session_id in video_subscribers:
        for queue in video_subscribers[session_id]:
            try:
                # Replace old frame - only keep latest
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except:
                        break
                queue.put_nowait(frame_jpeg)
            except asyncio.QueueFull:
                pass  # Drop if queue is full


async def publish_audio_to_subscribers(audio_data: bytes, webrtc_id: str = None):
    """Publish translated audio to HTTP stream subscribers."""
    # Publish to specific session subscribers
    if webrtc_id and webrtc_id in audio_subscribers:
        for queue in audio_subscribers[webrtc_id]:
            try:
                queue.put_nowait(audio_data)
            except asyncio.QueueFull:
                pass  # Drop if queue is full

    # Also publish to broadcast subscribers (any session)
    for queue in all_audio_subscribers:
        try:
            queue.put_nowait(audio_data)
        except asyncio.QueueFull:
            pass  # Drop if queue is full


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


@app.get("/audio/broadcast")
async def audio_broadcast_stream():
    """
    Broadcast audio stream - receives translated audio from any active session.

    This endpoint is useful for NDI/SRT/RTMP output where you want to capture
    translated audio regardless of the specific session ID.

    Usage:
        ffmpeg -f s16le -ar 24000 -ac 1 -i http://host/audio/broadcast -c:a aac -f mpegts srt://dest:port
    """
    async def generate_audio():
        queue = asyncio.Queue(maxsize=100)  # Buffer up to 100 chunks
        all_audio_subscribers.append(queue)

        try:
            while True:
                try:
                    audio_data = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield audio_data
                except asyncio.TimeoutError:
                    # Send silence to keep connection alive
                    yield b'\x00' * 4800  # 100ms of silence at 24kHz
        finally:
            all_audio_subscribers.remove(queue)

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


@app.get("/video/stream/{session_id}")
async def video_stream(session_id: str):
    """
    MJPEG video stream for remote viewers.

    Streams the source video as Motion JPEG for low-latency viewing.
    Can be embedded in <img> tags or used with video players.

    Usage:
        <img src="/video/stream/SESSION_ID">
        ffplay -f mjpeg http://host/video/stream/SESSION_ID
    """
    async def generate_mjpeg():
        queue = asyncio.Queue(maxsize=5)  # Small buffer for low latency
        if session_id not in video_subscribers:
            video_subscribers[session_id] = []
        video_subscribers[session_id].append(queue)

        try:
            while True:
                try:
                    jpeg_data = await asyncio.wait_for(queue.get(), timeout=30.0)
                    # MJPEG multipart format
                    yield b'--frame\r\n'
                    yield b'Content-Type: image/jpeg\r\n\r\n'
                    yield jpeg_data
                    yield b'\r\n'
                except asyncio.TimeoutError:
                    # Keep connection alive with empty boundary
                    yield b'--frame\r\n\r\n'
        finally:
            if session_id in video_subscribers:
                video_subscribers[session_id].remove(queue)

    return StreamingResponse(
        generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Access-Control-Allow-Origin": "*"
        }
    )


@app.get("/broadcast/info")
async def broadcast_info():
    """
    Information for broadcast engineers on how to integrate.
    """
    return JSONResponse({
        "service": "Zen Live Translate - Broadcast Integration",
        "endpoints": {
            "control_room_ui": f"{BASE_URL}/",
            "monitor_only": f"{BASE_URL}/monitor?autostart=1",
            "remote_view": f"{BASE_URL}/view?session={{SESSION_ID}}",
            "webrtc_offer": f"{BASE_URL}/webrtc/offer",
            "whip_ingest": f"{BASE_URL}/whip?src_language=Spanish&target_language=English",
            "whep_consume": f"{BASE_URL}/whep",
            "transcript_sse": f"{BASE_URL}/outputs?webrtc_id={{SESSION_ID}}",
            "video_mjpeg": f"{BASE_URL}/video/stream/{{SESSION_ID}}",
            "audio_pcm": f"{BASE_URL}/audio/stream/{{SESSION_ID}}",
            "audio_wav": f"{BASE_URL}/audio/wav/{{SESSION_ID}}",
            "audio_broadcast": f"{BASE_URL}/audio/broadcast",
            "api_status": f"{BASE_URL}/api/status",
            "api_sessions": f"{BASE_URL}/api/sessions"
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
            "note": "All examples use HTTPS - works through Cloudflare tunnel!",
            "ffplay_direct": f"ffplay -f s16le -ar 24000 -ac 1 {BASE_URL}/audio/stream/SESSION_ID",
            "ffplay_broadcast": f"ffplay -f s16le -ar 24000 -ac 1 {BASE_URL}/audio/broadcast",
            "ffmpeg_to_srt": f"ffmpeg -f s16le -ar 24000 -ac 1 -i {BASE_URL}/audio/broadcast -c:a aac -f mpegts 'srt://your-dest:port'",
            "ffmpeg_to_rtmp": f"ffmpeg -f s16le -ar 24000 -ac 1 -i {BASE_URL}/audio/broadcast -c:a aac -f flv rtmp://your-server/live/stream",
            "ffmpeg_to_ndi": f"ffmpeg -f s16le -ar 24000 -ac 1 -i {BASE_URL}/audio/broadcast -f libndi_newtek -clock_video true 'ZEN-LIVE-TRANSLATED'",
            "obs_browser": f"Add Browser Source with URL: {BASE_URL}/monitor?autostart=1",
            "vlc": f"vlc {BASE_URL}/audio/wav/SESSION_ID",
            "curl_test": f"curl -N {BASE_URL}/audio/broadcast | ffplay -f s16le -ar 24000 -ac 1 -"
        },
        "cloudflare_tunnel_notes": {
            "https_works": "All HTTP/HTTPS endpoints work through Cloudflare tunnel",
            "websocket_works": "WebSocket (wss://) works through Cloudflare tunnel",
            "srt_rtmp_input_needs_direct": "SRT/RTMP INPUT requires direct server access (not via tunnel)",
            "srt_rtmp_output_works": "SRT/RTMP OUTPUT works - ffmpeg pulls from HTTPS and pushes to your destination",
            "recommended_workflow": "Use WHIP for input (WebRTC over HTTPS), use /audio/broadcast for output"
        },
        "authentication": {
            "enabled": AUTH_ENABLED,
            "type": "HTTP Basic Auth" if AUTH_ENABLED else "None",
            "header": "Authorization: Basic <base64(username:password)>" if AUTH_ENABLED else None,
            "username": AUTH_USER if AUTH_ENABLED else None,
            "example_header": f"Authorization: Basic {base64.b64encode(f'{AUTH_USER}:{AUTH_PASS}'.encode()).decode()}" if AUTH_ENABLED else None,
            "curl_example": f"curl -u '{AUTH_USER}:{AUTH_PASS}' {BASE_URL}/" if AUTH_ENABLED else f"curl {BASE_URL}/",
            "websocket_example": f"wscat -c '{BASE_URL.replace('https', 'wss').replace('http', 'ws')}/v1/realtime' -H 'Authorization: Basic {base64.b64encode(f'{AUTH_USER}:{AUTH_PASS}'.encode()).decode()}'" if AUTH_ENABLED else f"wscat -c '{BASE_URL.replace('https', 'wss').replace('http', 'ws')}/v1/realtime'"
        },
        "session_management": {
            "description": "Each connection creates a unique session identified by webrtc_id/SESSION_ID",
            "how_to_get_session_id": {
                "webrtc": "Returned in response to /api/webrtc/offer as 'webrtc_id'",
                "whip": "Returned in Location header after POST to /whip",
                "example_response": {"webrtc_id": "abc123def456..."},
            },
            "using_session_id": {
                "transcripts": "GET /outputs?webrtc_id={your_session_id}",
                "audio_stream": "GET /audio/stream/{your_session_id}",
                "audio_wav": "GET /audio/wav/{your_session_id}",
                "note": "Replace {your_session_id} with actual session ID from connection response"
            }
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
# Streaming Input/Output Management (SRT, RTMP, NDI)
# =============================================================================

# Active ffmpeg processes for streaming
streaming_processes = {}  # session_id -> {"type": "srt_in"|"srt_out"|"rtmp_out", "process": subprocess.Popen}


def check_ffmpeg():
    """Check if ffmpeg is available."""
    return shutil.which("ffmpeg") is not None


class StreamingConfig(BaseModel):
    """Configuration for streaming input/output."""
    session_id: Optional[str] = None
    url: str
    src_language: str = "Spanish"
    target_language: str = "English"
    voice: str = "Cherry"


@app.get("/api/streaming/status")
async def streaming_status(_auth: bool = Depends(optional_auth)):
    """Get status of all active streaming processes."""
    ffmpeg_available = check_ffmpeg()

    active_streams = []
    for sid, info in streaming_processes.items():
        proc = info.get("process")
        active_streams.append({
            "session_id": sid,
            "type": info.get("type"),
            "url": info.get("url"),
            "running": proc.poll() is None if proc else False
        })

    return JSONResponse({
        "ffmpeg_available": ffmpeg_available,
        "active_streams": active_streams,
        "supported_inputs": ["srt", "rtmp"] if ffmpeg_available else [],
        "supported_outputs": ["srt", "rtmp", "ndi"] if ffmpeg_available else []
    })


@app.post("/api/streaming/output/start")
async def start_streaming_output(
    session_id: str,
    output_type: str,  # "srt", "rtmp", or "ndi"
    output_url: str,
    _auth: bool = Depends(optional_auth)
):
    """
    Start streaming translated audio to SRT/RTMP/NDI destination.

    This spawns an ffmpeg process that reads from /audio/broadcast and outputs
    to the specified destination.
    """
    if not check_ffmpeg():
        raise HTTPException(500, "ffmpeg not available - install ffmpeg to enable streaming")

    # Build audio input URL
    audio_url = f"http://localhost:{PORT}/audio/broadcast"

    # Build ffmpeg command based on output type
    if output_type == "srt":
        if not output_url.startswith("srt://"):
            raise HTTPException(400, "SRT URL must start with srt://")
        cmd = [
            "ffmpeg", "-re",
            "-f", "s16le", "-ar", "24000", "-ac", "1",
            "-i", audio_url,
            "-c:a", "aac", "-b:a", "128k",
            "-f", "mpegts",
            output_url
        ]
    elif output_type == "rtmp":
        if not output_url.startswith("rtmp://"):
            raise HTTPException(400, "RTMP URL must start with rtmp://")
        cmd = [
            "ffmpeg", "-re",
            "-f", "s16le", "-ar", "24000", "-ac", "1",
            "-i", audio_url,
            "-c:a", "aac", "-b:a", "128k",
            "-f", "flv",
            output_url
        ]
    elif output_type == "ndi":
        # NDI requires libndi - output_url is the NDI source name
        ndi_name = output_url if output_url else "ZEN-LIVE-TRANSLATED"
        cmd = [
            "ffmpeg", "-re",
            "-f", "s16le", "-ar", "24000", "-ac", "1",
            "-i", audio_url,
            "-f", "libndi_newtek",
            "-clock_video", "true",
            ndi_name
        ]
    else:
        raise HTTPException(400, f"Unsupported output type: {output_type}")

    # Start ffmpeg process
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        output_id = f"{output_type}_{session_id}_{secrets.token_hex(4)}"
        streaming_processes[output_id] = {
            "type": f"{output_type}_out",
            "url": output_url,
            "process": process,
            "session_id": session_id
        }

        print(f"📤 Started {output_type} output: {output_url} (PID: {process.pid})")

        return JSONResponse({
            "status": "started",
            "output_id": output_id,
            "output_type": output_type,
            "output_url": output_url,
            "pid": process.pid
        })

    except Exception as e:
        raise HTTPException(500, f"Failed to start ffmpeg: {str(e)}")


@app.delete("/api/streaming/output/{output_id}")
async def stop_streaming_output(output_id: str, _auth: bool = Depends(optional_auth)):
    """Stop a streaming output process."""
    if output_id not in streaming_processes:
        raise HTTPException(404, "Output stream not found")

    info = streaming_processes[output_id]
    proc = info.get("process")

    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    del streaming_processes[output_id]
    print(f"📤 Stopped output: {output_id}")

    return JSONResponse({"status": "stopped", "output_id": output_id})


@app.post("/api/streaming/input/start")
async def start_streaming_input(config: StreamingConfig, _auth: bool = Depends(optional_auth)):
    """
    Start ingesting from SRT/RTMP source for translation.

    Note: This is for server-side ingestion. The audio will be fed into a
    translation session. For browser-based input, use the WebRTC endpoints.
    """
    if not check_ffmpeg():
        raise HTTPException(500, "ffmpeg not available - install ffmpeg to enable streaming")

    url = config.url

    # Validate URL
    if url.startswith("srt://"):
        input_type = "srt"
    elif url.startswith("rtmp://"):
        input_type = "rtmp"
    else:
        raise HTTPException(400, "URL must start with srt:// or rtmp://")

    session_id = config.session_id or secrets.token_hex(16)

    # Build ffmpeg command to ingest and convert to PCM
    cmd = [
        "ffmpeg", "-re",
        "-i", url,
        "-c:a", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-f", "s16le",
        "pipe:1"
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        input_id = f"{input_type}_in_{session_id}"
        streaming_processes[input_id] = {
            "type": f"{input_type}_in",
            "url": url,
            "process": process,
            "session_id": session_id,
            "config": config.model_dump()
        }

        print(f"📥 Started {input_type} input: {url} (PID: {process.pid})")

        # TODO: Feed audio from process.stdout to a translation handler
        # This requires creating a handler instance and feeding audio chunks
        # For now, this sets up the process - full integration TBD

        return JSONResponse({
            "status": "started",
            "input_id": input_id,
            "input_type": input_type,
            "session_id": session_id,
            "input_url": url,
            "pid": process.pid,
            "note": "Audio ingestion started. Full translation integration in progress."
        })

    except Exception as e:
        raise HTTPException(500, f"Failed to start ffmpeg: {str(e)}")


@app.delete("/api/streaming/input/{input_id}")
async def stop_streaming_input(input_id: str, _auth: bool = Depends(optional_auth)):
    """Stop a streaming input process."""
    if input_id not in streaming_processes:
        raise HTTPException(404, "Input stream not found")

    info = streaming_processes[input_id]
    proc = info.get("process")

    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    del streaming_processes[input_id]
    print(f"📥 Stopped input: {input_id}")

    return JSONResponse({"status": "stopped", "input_id": input_id})


# =============================================================================
# WebSocket Proxy API - Proxies to Hanzo
# =============================================================================

# Full language support per Hanzo docs
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

# Full voice support per Hanzo docs
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
    WebSocket proxy to Hanzo

    This endpoint proxies all WebSocket messages to Hanzo's real-time
    translation API.

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
    if not HANZO_API_KEY:
        await websocket.close(code=4001, reason="HANZO_API_KEY not configured")
        return

    # Verify authentication before accepting
    if not await verify_websocket_auth(websocket):
        await websocket.close(code=4003, reason="Authentication required")
        return

    await websocket.accept()

    upstream_ws = None
    try:
        # Connect to Hanzo
        upstream_ws = await connect(
            TRANSLATE_API_URL,
            additional_headers={"Authorization": f"Bearer {HANZO_API_KEY}"},
            ssl=ssl_context
        )

        async def forward_to_upstream():
            """Forward client messages to Hanzo."""
            try:
                while True:
                    data = await websocket.receive_text()
                    await upstream_ws.send(data)
            except WebSocketDisconnect:
                pass
            except Exception as e:
                print(f"Client->Upstream error: {e}")

        async def forward_to_client():
            """Forward Hanzo responses to client."""
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
    Full API specification for Zen Live.

    This documents the WebSocket protocol for real-time translation.
    """
    ws_url = BASE_URL.replace("https", "wss").replace("http", "ws")

    return JSONResponse({
        "name": "Zen Live Translation API",
        "version": "1.0.0",
        "description": "Real-time audio/video translation powered by Zen Live",
        "base_url": BASE_URL,
        "websocket_endpoint": f"{ws_url}/v1/realtime",
        "protocol": "WebSocket",
        "authentication": {
            "http_basic": {
                "enabled": AUTH_ENABLED,
                "type": "HTTP Basic Auth" if AUTH_ENABLED else "None",
                "username": AUTH_USER if AUTH_ENABLED else "Not required",
                "header_format": "Authorization: Basic <base64(username:password)>",
                "example": f"Authorization: Basic {base64.b64encode(f'{AUTH_USER}:{AUTH_PASS}'.encode()).decode()}" if AUTH_ENABLED else "No auth required",
                "curl": f"curl -u '{AUTH_USER}:{AUTH_PASS}' {BASE_URL}/api/status" if AUTH_ENABLED else f"curl {BASE_URL}/api/status",
                "websocket": f"wscat -c '{ws_url}/v1/realtime' -H 'Authorization: Basic {base64.b64encode(f'{AUTH_USER}:{AUTH_PASS}'.encode()).decode()}'" if AUTH_ENABLED else f"wscat -c '{ws_url}/v1/realtime'"
            },
            "api_key": {
                "type": "Bearer Token (server-side)",
                "note": "API key for Hanzo is configured server-side via HANZO_API_KEY env var",
                "configured": bool(HANZO_API_KEY)
            }
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


# =============================================================================
# WebSocket Proxy for ASR
# =============================================================================


@app.websocket("/v1/asr")
async def asr_websocket_proxy(websocket: WebSocket):
    """
    WebSocket proxy to Hanzo.

    This endpoint proxies ASR requests to Hanzo's real-time ASR API,
    keeping the API key secure on the server side.

    Client sends:
    - session.update: Configure ASR session (language, model)
    - input_audio_buffer.append: Send Base64-encoded PCM16 audio

    Server returns:
    - session.created/updated: Session confirmations
    - conversation.item.input_audio_transcription.text: Interim transcription
    - conversation.item.input_audio_transcription.completed: Final transcription
    - input_audio_buffer.speech_started/stopped: VAD events
    - error: Error details
    """
    if not HANZO_API_KEY:
        await websocket.close(code=4001, reason="HANZO_API_KEY not configured")
        return

    # Verify authentication before accepting
    if not await verify_websocket_auth(websocket):
        await websocket.close(code=4003, reason="Authentication required")
        return

    await websocket.accept()

    upstream_ws = None
    try:
        # Connect to Hanzo ASR endpoint
        upstream_ws = await connect(
            ASR_API_URL,
            additional_headers={"Authorization": f"Bearer {HANZO_API_KEY}"},
            ssl=ssl_context
        )
        print("🎙️ ASR WebSocket proxy connected to Hanzo")

        async def forward_to_upstream():
            """Forward client messages to Hanzo ASR."""
            try:
                while True:
                    data = await websocket.receive_text()
                    await upstream_ws.send(data)
            except WebSocketDisconnect:
                pass
            except Exception as e:
                print(f"ASR Client->Upstream error: {e}")

        async def forward_to_client():
            """Forward Hanzo ASR responses to client."""
            try:
                async for message in upstream_ws:
                    await websocket.send_text(message)
            except Exception as e:
                print(f"ASR Upstream->Client error: {e}")

        # Run both directions concurrently
        await asyncio.gather(
            forward_to_upstream(),
            forward_to_client(),
            return_exceptions=True
        )

    except Exception as e:
        print(f"ASR WebSocket proxy error: {e}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "error": {"message": str(e), "type": "asr_proxy_error"}
            }))
        except:
            pass
    finally:
        if upstream_ws:
            await upstream_ws.close()
        print("🎙️ ASR WebSocket proxy disconnected")


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
        "default": "Nofish",
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

    # Display startup banner with proper URLs
    local_url = f"http://localhost:{PORT}"
    prod_note = f" ({BASE_URL})" if "hanzo.ai" in BASE_URL else ""

    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║                       ZEN LIVE                            ║
    ║         Real-time Speech Translation Service              ║
    ║                   Powered by Hanzo AI                     ║
    ╠═══════════════════════════════════════════════════════════╣
    ║  Control Room:  {local_url:<42} ║
    ║  Monitor View:  {local_url}/monitor{' '*(36-len(str(PORT)))} ║
    ║  API Docs:      {local_url}/docs{' '*(39-len(str(PORT)))} ║
    ║  Broadcast:     {local_url}/broadcast/info{' '*(27-len(str(PORT)))} ║
    ║                                                           ║
    ║  Production:    {BASE_URL:<42} ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    if mode == "UI":
        demo = enhance_ui()
        demo.launch(server_port=PORT)
    elif mode == "PHONE":
        stream.fastphone(host="0.0.0.0", port=PORT)
    else:
        uvicorn.run(app, host="0.0.0.0", port=PORT)
