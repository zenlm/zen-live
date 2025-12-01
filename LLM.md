# Zen Live

Real-time speech translation for broadcast news monitoring.

## Project Context

This is part of the [Zen AI model family](https://github.com/zenlm/zen), providing a hosted translation service via [Hanzo Node](https://github.com/hanzoai/hanzo-node) infrastructure.

## Architecture

```
Control Room Browser
        │
        │ WebRTC (audio/video)
        ▼
    Zen Live (FastRTC/FastAPI)
        │
        ├──► Hanzo Node API (recommended)
        │         │
        │         ▼
        │    Qwen3 LiveTranslate
        │
        └──► Direct DashScope API (fallback)
                  │
                  ▼
             Qwen3 LiveTranslate
```

## Backend Options

1. **Hanzo Node** (recommended): `HANZO_NODE_URL=http://host:9550`
2. **DashScope Direct**: `API_KEY=xxx`
3. **Zen Omni Local** (future): `ZEN_OMNI_PATH=/path/to/model`

## Files

| File | Purpose |
|------|---------|
| `app.py` | Main FastAPI/FastRTC application |
| `index.html` | Control room web portal |
| `monitor.html` | Simplified broadcast display |
| `requirements.txt` | Python dependencies |
| `README.md` | User documentation |
| `LLM.md` | AI assistant context (this file) |

## Key Endpoints

- `/` - Control room UI
- `/monitor` - Broadcast monitor view
- `/api/status` - Health check
- `/broadcast/info` - Engineer integration guide
- `/audio/stream/{id}` - PCM16 audio stream
- `/outputs?webrtc_id={id}` - SSE transcripts

## Default Configuration

- **Source**: Spanish (news monitoring use case)
- **Target**: English
- **Voice**: Cherry
- **Audio**: PCM16, 24kHz, mono
- **Latency**: ~200-500ms

## Integration with Hanzo Node

When `HANZO_NODE_URL` is set, Zen Live can:
1. Query configured LLM providers from hanzo-node
2. Use node's API key management
3. Leverage node's monitoring/logging

Future: Native integration with zen-omni model for offline operation.

## Development Notes

- WebRTC via FastRTC library (Gradio ecosystem)
- CORS enabled for cross-origin control room access
- Audio subscribers pattern for broadcast streaming
- SSE for real-time transcript delivery

## Links

- GitHub: https://github.com/zenlm/zen-live
- Hanzo Node: https://github.com/hanzoai/hanzo-node
- Zen Models: https://github.com/zenlm/zen
- Hanzo AI: https://hanzo.ai
