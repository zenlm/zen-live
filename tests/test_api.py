"""
Integration tests for API endpoints.

Tests cover:
- Health check endpoint
- WebRTC offer/answer flow
- SSE transcript streaming
- Audio/video streaming endpoints
- Session management
"""

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.asyncio


class TestHealthCheck:
    """Tests for health check endpoint."""

    async def test_status_endpoint_returns_healthy(self):
        """Verify /api/status returns healthy status."""
        with patch.dict(
            "os.environ",
            {
                "TRANSLATE_API_URL": "wss://test.example.com/api",
                "HANZO_API_KEY": "test-key",
                "BASE_URL": "https://test.zen-live.hanzo.ai",
            },
        ):
            from fastapi.testclient import TestClient

            from app import app

            client = TestClient(app)
            response = client.get("/api/status")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["service"] == "zen-live-translate"
            assert "supported_languages" in data

    async def test_status_includes_language_support(self):
        """Verify status includes supported languages."""
        with patch.dict(
            "os.environ",
            {
                "TRANSLATE_API_URL": "wss://test.example.com/api",
                "HANZO_API_KEY": "test-key",
                "BASE_URL": "https://test.zen-live.hanzo.ai",
            },
        ):
            from fastapi.testclient import TestClient

            from app import app

            client = TestClient(app)
            response = client.get("/api/status")

            data = response.json()
            langs = data["supported_languages"]

            assert "source" in langs
            assert "target" in langs
            assert "Spanish" in langs["source"]
            assert "English" in langs["target"]

    async def test_status_includes_voices(self):
        """Verify status includes available voices."""
        with patch.dict(
            "os.environ",
            {
                "TRANSLATE_API_URL": "wss://test.example.com/api",
                "HANZO_API_KEY": "test-key",
                "BASE_URL": "https://test.zen-live.hanzo.ai",
            },
        ):
            from fastapi.testclient import TestClient

            from app import app

            client = TestClient(app)
            response = client.get("/api/status")

            data = response.json()
            assert "voices" in data
            assert "Cherry" in data["voices"]


class TestSessionManagement:
    """Tests for session creation and management."""

    async def test_session_config_endpoint_exists(self):
        """Verify session-related endpoints are accessible."""
        with patch.dict(
            "os.environ",
            {
                "TRANSLATE_API_URL": "wss://test.example.com/api",
                "HANZO_API_KEY": "test-key",
                "BASE_URL": "https://test.zen-live.hanzo.ai",
            },
        ):
            from fastapi.testclient import TestClient

            from app import app

            client = TestClient(app)
            # Status endpoint should work and show session info
            response = client.get("/api/status")
            assert response.status_code == 200
            data = response.json()
            assert "active_sessions" in data


class TestWebRTCFlow:
    """Tests for WebRTC signaling."""

    async def test_webrtc_offer_requires_sdp(self):
        """Verify WebRTC offer requires SDP."""
        with patch.dict(
            "os.environ",
            {
                "TRANSLATE_API_URL": "wss://test.example.com/api",
                "HANZO_API_KEY": "test-key",
                "BASE_URL": "https://test.zen-live.hanzo.ai",
            },
        ):
            from fastapi.testclient import TestClient

            from app import app

            client = TestClient(app)

            # Missing SDP should fail
            response = client.post("/webrtc/offer", json={})
            assert response.status_code in [400, 422]


class TestTranscriptStreaming:
    """Tests for SSE transcript streaming."""

    async def test_outputs_endpoint_exists(self):
        """Verify outputs SSE endpoint exists."""
        with patch.dict(
            "os.environ",
            {
                "TRANSLATE_API_URL": "wss://test.example.com/api",
                "HANZO_API_KEY": "test-key",
                "BASE_URL": "https://test.zen-live.hanzo.ai",
            },
        ):
            from fastapi.testclient import TestClient

            from app import app

            client = TestClient(app)

            # Should require webrtc_id parameter
            response = client.get("/outputs")
            # Either 422 (missing param) or 200 with empty stream
            assert response.status_code in [200, 422]


class TestAudioStreaming:
    """Tests for audio streaming endpoints."""

    async def test_audio_stream_endpoint_exists(self):
        """Verify audio stream endpoint exists."""
        with patch.dict(
            "os.environ",
            {
                "TRANSLATE_API_URL": "wss://test.example.com/api",
                "HANZO_API_KEY": "test-key",
                "BASE_URL": "https://test.zen-live.hanzo.ai",
            },
        ):
            from fastapi.testclient import TestClient

            from app import app

            client = TestClient(app)

            response = client.get("/api/audio/test-session")
            # Should return audio stream or 404 if no session
            assert response.status_code in [200, 404]


class TestVideoStreaming:
    """Tests for video streaming endpoints."""

    async def test_video_stream_endpoint_exists(self):
        """Verify video stream endpoint exists."""
        with patch.dict(
            "os.environ",
            {
                "TRANSLATE_API_URL": "wss://test.example.com/api",
                "HANZO_API_KEY": "test-key",
                "BASE_URL": "https://test.zen-live.hanzo.ai",
            },
        ):
            from fastapi.testclient import TestClient

            from app import app

            client = TestClient(app)

            response = client.get("/api/video/test-session")
            # Should return video stream or 404 if no session
            assert response.status_code in [200, 404]


class TestStaticPages:
    """Tests for static page endpoints."""

    async def test_index_page_loads(self):
        """Verify index page loads (with auth if enabled)."""
        with patch.dict(
            "os.environ",
            {
                "TRANSLATE_API_URL": "wss://test.example.com/api",
                "HANZO_API_KEY": "test-key",
                "BASE_URL": "https://test.zen-live.hanzo.ai",
            },
        ):
            from fastapi.testclient import TestClient

            from app import AUTH_ENABLED, AUTH_PASS, AUTH_USER, app

            client = TestClient(app)

            if AUTH_ENABLED:
                # Provide auth credentials
                import base64

                credentials = base64.b64encode(f"{AUTH_USER}:{AUTH_PASS}".encode()).decode()
                response = client.get("/", headers={"Authorization": f"Basic {credentials}"})
            else:
                response = client.get("/")

            assert response.status_code == 200
            assert "text/html" in response.headers.get("content-type", "")

    async def test_monitor_page_loads(self):
        """Verify monitor page loads (with auth if enabled)."""
        with patch.dict(
            "os.environ",
            {
                "TRANSLATE_API_URL": "wss://test.example.com/api",
                "HANZO_API_KEY": "test-key",
                "BASE_URL": "https://test.zen-live.hanzo.ai",
            },
        ):
            from fastapi.testclient import TestClient

            from app import AUTH_ENABLED, AUTH_PASS, AUTH_USER, app

            client = TestClient(app)

            if AUTH_ENABLED:
                # Provide auth credentials
                import base64

                credentials = base64.b64encode(f"{AUTH_USER}:{AUTH_PASS}".encode()).decode()
                response = client.get("/monitor", headers={"Authorization": f"Basic {credentials}"})
            else:
                response = client.get("/monitor")

            assert response.status_code == 200
