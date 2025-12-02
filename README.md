---
title: Zen Live
emoji: 🎙️
colorFrom: purple
colorTo: red
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Real-time speech translation for broadcast news
---

# Zen Live

Real-time speech translation for broadcast news monitoring. Powered by [Hanzo AI](https://hanzo.ai).

## Overview

Zen Live is a low-latency simultaneous translation service designed for news control rooms. It takes audio/video input in one language and outputs translated audio + captions in real-time via WebRTC.

**Default configuration: Spanish → English** (configurable for 18+ language pairs)

## Features

- **~200-500ms end-to-end latency** - Suitable for live broadcast monitoring
- **WebRTC streaming** - Simple browser-based consumption for control rooms
- **Multiple output formats** - WebRTC, HTTP audio streams (PCM/WAV), SSE transcripts
- **Broadcast integration** - Convert to SRT/RTMP/NDI via ffmpeg
- **Control room UI** - Professional interface with transcript logging
- **Monitor mode** - Clean fullscreen view for broadcast displays

## Quick Start

```bash
# Clone the repository
git clone https://github.com/zenlm/zen-live.git
cd zen-live

# Install dependencies
pip install -r requirements.txt

# Configure backend (choose one):
export HANZO_NODE_URL=http://localhost:9550   # Recommended: Hanzo Node
# or
export API_KEY=your_hanzo_api_key         # Direct Hanzo API

# Run
python app.py
```

Open http://localhost:8000 in your browser.

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `/` | Control room web portal |
| `/monitor` | Simplified broadcast display view |
| `/monitor?autostart=1` | Auto-start for OBS/video walls |
| `/api/status` | Service health check |
| `/api/sessions` | Active session list |
| `/broadcast/info` | Integration guide for engineers |
| `/docs` | **OpenAPI (Swagger) documentation** |
| `/redoc` | ReDoc API documentation |
| `/openapi.json` | OpenAPI JSON spec for code generation |
| `/whip` | WHIP endpoint for broadcaster ingestion |
| `/whep` | WHEP endpoint for WebRTC consumption |
| `/outputs?webrtc_id=ID` | SSE transcript stream |
| `/audio/stream/ID` | Raw PCM16 audio (24kHz) |
| `/audio/wav/ID` | WAV-wrapped audio stream |

## API Documentation

Full OpenAPI/Swagger documentation is available at `/docs` when the server is running.

- **Interactive Docs:** `http://your-server:8000/docs`
- **ReDoc:** `http://your-server:8000/redoc`
- **OpenAPI JSON:** `http://your-server:8000/openapi.json`

Use the OpenAPI spec to generate client SDKs in any language.

## Backend Options

### 1. Hanzo Node (Recommended)

Connect to a [Hanzo Node](https://github.com/hanzoai/hanzo-node) instance for managed translation infrastructure.

```bash
export HANZO_NODE_URL=http://your-hanzo-node:9550
```

### 2. Direct Hanzo API

Use Hanzo Zen Live API directly (requires API key).

```bash
export API_KEY=your_hanzo_key
```

### 3. Local Zen Omni Model (Coming Soon)

Run completely offline with local Zen Omni model.

```bash
export ZEN_OMNI_PATH=/path/to/zen-omni
```

## Control Room Usage

### For News Teams

1. Open **http://server:8000** in Chrome/Edge
2. Select source language (default: Spanish)
3. Select target language (default: English)
4. Click **Start** to begin translation
5. Listen to translated audio through speakers/headphones
6. View live captions on screen

### For Broadcast Engineers

**OBS Integration:**
```
Add Browser Source → http://server:8000/monitor?autostart=1
```

**Direct Audio Monitoring:**
```bash
ffplay -f s16le -ar 24000 -ac 1 http://server:8000/audio/stream/SESSION_ID
```

**Convert to SRT:**
```bash
ffmpeg -f s16le -ar 24000 -ac 1 -i http://server:8000/audio/stream/SESSION_ID \
       -c:a aac -f mpegts 'srt://broadcast-server:9000'
```

**Convert to RTMP:**
```bash
ffmpeg -f s16le -ar 24000 -ac 1 -i http://server:8000/audio/stream/SESSION_ID \
       -c:a aac -f flv rtmp://server/live/translation
```

## Supported Languages

### Source Languages
Spanish, English, Chinese, Portuguese, French, German, Russian, Italian, Korean, Japanese, Cantonese, Indonesian, Vietnamese, Thai, Arabic, Hindi, Greek, Turkish

### Target Languages
English, Chinese, Russian, French, German, Portuguese, Spanish, Italian, Korean, Japanese, Cantonese, Indonesian, Vietnamese, Thai, Arabic

## Audio Format

- **Encoding:** PCM16 signed little-endian
- **Sample Rate:** 24,000 Hz
- **Channels:** Mono (1)
- **Bits per Sample:** 16

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HANZO_NODE_URL` | Hanzo Node backend URL | - |
| `API_KEY` | Hanzo API key (fallback) | - |
| `ZEN_OMNI_PATH` | Local model path | - |
| `ZEN_LIVE_USER` | HTTP Basic Auth username (optional) | - |
| `ZEN_LIVE_PASS` | HTTP Basic Auth password (optional) | - |
| `WHIP_ENABLED` | Enable WHIP ingestion endpoint | `true` |
| `PORT` | Server port | 8000 |
| `MODE` | `UI` for Gradio, `PHONE` for FastPhone | - |

## Authentication

When both `ZEN_LIVE_USER` and `ZEN_LIVE_PASS` are set, HTTP Basic Authentication is required for the control room UI and monitor pages. API endpoints remain accessible for integration.

```bash
export ZEN_LIVE_USER=operator@news.com
export ZEN_LIVE_PASS=your_secure_password
```

## Architecture

```
┌─────────────────┐      WebRTC       ┌──────────────────┐
│  Control Room   │◄─────────────────►│    Zen Live      │
│   Browser UI    │                   │  (FastRTC/ASGI)  │
└─────────────────┘                   └────────┬─────────┘
                                               │
                                    ┌──────────┴──────────┐
                                    │                     │
                              ┌─────▼─────┐        ┌──────▼─────┐
                              │   Hanzo   │        │    Hanzo   │
                              │   Node    │        │    API     │
                              └───────────┘        └────────────┘
                                    │                     │
                                    └──────────┬──────────┘
                                               │
                                    ┌──────────▼────────────┐
                                    │  Zen Omni + Zen Live  │
                                    │     (Backend)         │
                                    └───────────────────────┘
```

## Related Projects

- [Hanzo Node](https://github.com/hanzoai/node) - AI infrastructure node
- [Zen AI Models](https://github.com/zenlm/zen) - Zen model family
- [Zen Omni](https://huggingface.co/zenlm/zen-omni) - Multimodal model

## License

Apache 2.0

## Links

- **Zen LM:** https://zenlm.org
- **Hanzo AI:** https://hanzo.ai
- **HuggingFace:** https://huggingface.co/zenlm
