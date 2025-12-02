# Zen Live Integration Guide

## Overview
Zen Live provides real-time speech translation for broadcast and streaming applications. Users send audio/video to our service and receive translated audio and transcripts back.

## Authentication
```bash
Username: admin
Password: ZenLiveRealtime2025!
Base64: YWRtaW46WmVuTGl2ZVJlYWx0aW1lMjAyNSE=
```

## Integration Methods

### 1. Web UI (Browser-Based)
**URL:** https://zen-live.hanzo.ai/

Users access the control room UI to:
1. Select **Source Language** (what's being spoken)
2. Select **Target Language** (translate to)
3. Select **Voice** (TTS voice for output)
4. Click **Start** to begin translation

The browser handles WebRTC automatically.

### 2. WebRTC API (Programmatic)
**Endpoint:** `POST https://zen-live.hanzo.ai/api/webrtc/offer`

```javascript
// Send WebRTC offer with configuration
const response = await fetch('https://zen-live.hanzo.ai/api/webrtc/offer', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Basic YWRtaW46WmVuTGl2ZVJlYWx0aW1lMjAyNSE='
    },
    body: JSON.stringify({
        sdp: localDescription.sdp,
        type: localDescription.type,
        src_language: "Spanish",     // Input language
        target_language: "English",   // Output language
        voice: "Cherry"              // TTS voice
    })
});

const answer = await response.json();
// Use answer.sdp to complete WebRTC connection
```

### 3. WHIP Protocol (Broadcaster Ingestion)
**Endpoint:** `POST https://zen-live.hanzo.ai/whip`

For professional broadcast equipment and encoders:

```bash
# FFmpeg example
ffmpeg -re -i input.mp4 \
    -c:v libvpx -c:a opus \
    -f webrtc \
    "https://admin:ZenLiveRealtime2025!@zen-live.hanzo.ai/whip?src_language=Spanish&target_language=English&voice=Cherry"

# GStreamer example
gst-launch-1.0 audiotestsrc ! opusenc ! \
    webrtcsink signaller::uri="https://admin:ZenLiveRealtime2025!@zen-live.hanzo.ai/whip"
```

### 4. WebSocket API (Direct Connection)
**Endpoint:** `wss://zen-live.hanzo.ai/v1/realtime`

```javascript
// Connect with authentication
const ws = new WebSocket('wss://zen-live.hanzo.ai/v1/realtime');
ws.addEventListener('open', () => {
    // Configure session
    ws.send(JSON.stringify({
        type: "session.update",
        session: {
            modalities: ["text", "audio"],
            voice: "Cherry",
            input_audio_format: "pcm16",
            output_audio_format: "pcm16",
            input_audio_transcription: {
                language: "es"  // Spanish input
            },
            translation: {
                language: "en"  // English output
            }
        }
    }));
});

// Send audio data
ws.send(JSON.stringify({
    type: "input_audio_buffer.append",
    audio: base64EncodedPCM16Audio
}));

// Receive translations
ws.addEventListener('message', (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'response.audio.delta') {
        // Translated audio (base64 PCM)
        const audioData = atob(data.delta);
    }
    if (data.type === 'response.audio_transcript.done') {
        // Final transcript
        console.log(data.transcript);
    }
});
```

## Data Flow

```
1. INPUT CONFIGURATION
   ├── Source Language (Spanish, English, Chinese, etc.)
   ├── Target Language (English, Spanish, Chinese, etc.)
   └── Voice (Cherry, Nofish, Jada, etc.)

2. SEND AUDIO/VIDEO
   ├── WebRTC (browser/app)
   ├── WHIP (broadcast equipment)
   └── WebSocket (direct API)

3. RECEIVE OUTPUT
   ├── Translated Audio (PCM16/24, 24kHz)
   ├── Source Transcript (what was said)
   └── Target Transcript (translation)
```

## Output Consumption

### Audio Streams
```bash
# Raw PCM audio stream
curl -u 'admin:ZenLiveRealtime2025!' \
     https://zen-live.hanzo.ai/audio/stream/{SESSION_ID}

# WAV-wrapped audio
curl -u 'admin:ZenLiveRealtime2025!' \
     https://zen-live.hanzo.ai/audio/wav/{SESSION_ID}
```

### Transcript Stream (SSE)
```bash
# Server-sent events for real-time transcripts
curl -u 'admin:ZenLiveRealtime2025!' \
     https://zen-live.hanzo.ai/outputs?webrtc_id={SESSION_ID}
```

## Configuration Parameters

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `src_language` | Spanish, English, Chinese, Russian, French, German, Portuguese, Italian, Korean, Japanese, etc. | Spanish | Input language |
| `target_language` | English, Spanish, Chinese, Russian, French, German, Portuguese, Italian, Korean, Japanese, etc. | English | Output language |
| `voice` | Cherry, Nofish, Jada, Dylan, Sunny, Peter, Kiki, Eric | Cherry | TTS voice |
| `input_audio_format` | pcm16 | pcm16 | Input audio encoding |
| `output_audio_format` | pcm16, pcm24 | pcm16 | Output audio encoding |

## Quick Start Examples

### Browser/JavaScript
```html
<button onclick="startTranslation()">Start Translation</button>

<script>
async function startTranslation() {
    // Get user media
    const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
        video: true
    });

    // Create peer connection
    const pc = new RTCPeerConnection();
    stream.getTracks().forEach(track => pc.addTrack(track, stream));

    // Create offer
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    // Send to Zen Live
    const response = await fetch('https://zen-live.hanzo.ai/api/webrtc/offer', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Basic YWRtaW46WmVuTGl2ZVJlYWx0aW1lMjAyNSE='
        },
        body: JSON.stringify({
            sdp: offer.sdp,
            type: offer.type,
            src_language: "Spanish",
            target_language: "English",
            voice: "Cherry"
        })
    });

    const answer = await response.json();
    await pc.setRemoteDescription(new RTCSessionDescription(answer));

    // Handle incoming translations
    pc.addEventListener('track', (event) => {
        // Play translated audio
        const audio = new Audio();
        audio.srcObject = event.streams[0];
        audio.play();
    });
}
</script>
```

### Python Client
```python
import asyncio
import websockets
import base64
import json

async def translate_audio():
    auth = base64.b64encode(b"admin:ZenLiveRealtime2025!").decode()
    headers = {"Authorization": f"Basic {auth}"}

    async with websockets.connect(
        "wss://zen-live.hanzo.ai/v1/realtime",
        extra_headers=headers
    ) as ws:
        # Configure session
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "voice": "Cherry",
                "input_audio_transcription": {"language": "es"},
                "translation": {"language": "en"}
            }
        }))

        # Send audio (you'd get this from microphone/file)
        with open("audio.pcm", "rb") as f:
            audio_data = f.read()
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio_data).decode()
            }))

        # Receive translations
        async for message in ws:
            data = json.loads(message)
            if data["type"] == "response.audio_transcript.done":
                print(f"Translation: {data['transcript']}")

asyncio.run(translate_audio())
```

### cURL Testing
```bash
# Test API status
curl -u 'admin:ZenLiveRealtime2025!' https://zen-live.hanzo.ai/api/status

# Get supported languages
curl -u 'admin:ZenLiveRealtime2025!' https://zen-live.hanzo.ai/api/languages

# Get available voices
curl -u 'admin:ZenLiveRealtime2025!' https://zen-live.hanzo.ai/api/voices
```

## Support

For integration help, visit:
- API Docs: https://zen-live.hanzo.ai/docs
- API Spec: https://zen-live.hanzo.ai/api/spec
- Broadcast Info: https://zen-live.hanzo.ai/broadcast/info
- GitHub: https://github.com/zenlm/zen-live