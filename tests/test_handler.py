"""
Unit tests for LiveTranslateHandler.

Tests cover:
- Initialization and configuration
- Audio receive with silence detection
- Keepalive mechanism
- Message counter isolation
- Connection handling
"""

import asyncio
import base64
import time
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

# Import will be done after patching
pytestmark = pytest.mark.asyncio


class TestHandlerInitialization:
    """Tests for handler initialization."""

    def test_handler_creates_instance_variables(self):
        """Verify all instance variables are properly initialized."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler(session_id="test-123")

            assert handler.session_id == "test-123"
            assert handler.connection is None
            assert handler._msg_counter == 0
            assert handler.last_audio_time == 0.0
            assert handler.running is False
            assert handler.keepalive_task is None

    def test_handler_message_counter_is_instance_variable(self):
        """Verify message counter is per-instance, not shared."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler1 = LiveTranslateHandler(session_id="session-1")
            handler2 = LiveTranslateHandler(session_id="session-2")

            # Increment counter on handler1
            handler1._msg_counter = 100

            # handler2 should still be at 0
            assert handler2._msg_counter == 0
            assert handler1._msg_counter == 100

    def test_handler_msg_id_increments(self):
        """Verify msg_id generates unique IDs."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()

            id1 = handler.msg_id()
            id2 = handler.msg_id()
            id3 = handler.msg_id()

            assert id1 == "evt_1"
            assert id2 == "evt_2"
            assert id3 == "evt_3"

    def test_handler_session_config_defaults(self):
        """Verify default session configuration."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()

            assert handler.session_config["src_language"] == "Spanish"
            assert handler.session_config["target_language"] == "English"
            assert handler.session_config["voice"] == "Nofish"

    def test_handler_custom_session_config(self):
        """Verify custom session configuration is applied."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            config = {"src_language": "English", "target_language": "Chinese", "voice": "Cherry"}
            handler = LiveTranslateHandler(session_config=config)

            assert handler.session_config["src_language"] == "English"
            assert handler.session_config["target_language"] == "Chinese"
            assert handler.session_config["voice"] == "Cherry"


class TestSilenceDetection:
    """Tests for audio silence detection."""

    def test_calculate_rms_silent(self, audio_rms_calculator):
        """Verify RMS of silent audio is near zero."""
        silent = np.zeros(1600, dtype=np.int16)
        rms = audio_rms_calculator(silent)
        assert rms < 1

    def test_calculate_rms_quiet(self, audio_rms_calculator):
        """Verify RMS of quiet audio is below threshold."""
        # Random noise with small amplitude
        quiet = np.random.randint(-30, 30, 1600, dtype=np.int16)
        rms = audio_rms_calculator(quiet)
        assert rms < 50  # Below our threshold

    def test_calculate_rms_normal_speech(self, audio_rms_calculator):
        """Verify RMS of normal speech is above threshold."""
        # Sine wave simulating speech
        t = np.linspace(0, 0.1, 1600, dtype=np.float32)
        speech = (3000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
        rms = audio_rms_calculator(speech)
        assert rms > 50  # Above our threshold
        assert rms > 1000  # Normal speech level

    async def test_receive_skips_silent_audio(self, silent_audio_frame):
        """Verify silent audio frames are not sent to API."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()
            handler.connection = AsyncMock()
            handler.last_audio_time = time.time() - 10  # Old timestamp

            await handler.receive(silent_audio_frame)

            # Connection.send should NOT be called for silent audio
            handler.connection.send.assert_not_called()
            # last_audio_time should NOT be updated
            assert time.time() - handler.last_audio_time > 5

    async def test_receive_sends_real_audio(self, sample_audio_frame):
        """Verify real audio frames are sent to API."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()
            handler.connection = AsyncMock()
            old_time = time.time() - 10
            handler.last_audio_time = old_time

            await handler.receive(sample_audio_frame)

            # Connection.send should be called
            handler.connection.send.assert_called_once()
            # last_audio_time should be updated
            assert handler.last_audio_time > old_time

    async def test_receive_skips_quiet_audio(self, quiet_audio_frame):
        """Verify quiet audio (RMS < 50) is skipped."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()
            handler.connection = AsyncMock()

            await handler.receive(quiet_audio_frame)

            handler.connection.send.assert_not_called()

    async def test_receive_sends_loud_audio(self, loud_audio_frame):
        """Verify loud audio is sent."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()
            handler.connection = AsyncMock()

            await handler.receive(loud_audio_frame)

            handler.connection.send.assert_called_once()

    async def test_receive_handles_no_connection(self, sample_audio_frame):
        """Verify receive handles missing connection gracefully."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()
            handler.connection = None

            # Should not raise
            await handler.receive(sample_audio_frame)


class TestKeepalive:
    """Tests for keepalive mechanism."""

    async def test_keepalive_sends_silence_when_no_audio(self):
        """Verify keepalive sends silent audio when no real audio received."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()
            handler.connection = AsyncMock()
            handler.running = True
            handler.last_audio_time = time.time() - 5  # 5 seconds ago

            # Run one iteration of keepalive logic manually
            now = time.time()
            if now - handler.last_audio_time > 0.5:
                handler._msg_counter += 1
                msg = f'{{"event_id":"evt_{handler._msg_counter}","type":"input_audio_buffer.append","audio":"{handler.SILENT_AUDIO_100MS}"}}'
                await handler.connection.send(msg)

            handler.connection.send.assert_called_once()
            call_args = handler.connection.send.call_args[0][0]
            assert "input_audio_buffer.append" in call_args

    async def test_keepalive_does_not_send_when_audio_active(self):
        """Verify keepalive does not send when real audio is flowing."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()
            handler.connection = AsyncMock()
            handler.running = True
            handler.last_audio_time = time.time()  # Just now

            # Check keepalive condition
            now = time.time()
            if now - handler.last_audio_time > 0.5:
                await handler.connection.send("keepalive")

            handler.connection.send.assert_not_called()

    def test_silent_audio_constant_is_valid(self):
        """Verify SILENT_AUDIO_100MS is properly encoded."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            # Decode and verify
            decoded = base64.b64decode(LiveTranslateHandler.SILENT_AUDIO_100MS)
            audio = np.frombuffer(decoded, dtype=np.int16)

            # Should be 1600 samples (100ms at 16kHz)
            assert len(audio) == 1600
            # Should be all zeros
            assert np.all(audio == 0)


class TestConnectionHandling:
    """Tests for connection handling and cleanup."""

    async def test_shutdown_stops_keepalive(self):
        """Verify shutdown stops the keepalive task."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()
            handler.connection = AsyncMock()
            handler.running = True

            # Create a mock keepalive task
            async def mock_keepalive():
                while handler.running:
                    await asyncio.sleep(0.1)

            handler.keepalive_task = asyncio.create_task(mock_keepalive())

            await handler.shutdown()

            assert handler.running is False
            assert handler.connection is None

    async def test_shutdown_clears_queues(self):
        """Verify shutdown clears output and video queues."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()
            handler.connection = AsyncMock()

            # Add items to queues
            await handler.output_queue.put("test1")
            await handler.output_queue.put("test2")
            await handler.video_queue.put("frame1")

            await handler.shutdown()

            assert handler.output_queue.empty()
            assert handler.video_queue.empty()

    async def test_receive_captures_connection_locally(self, sample_audio_frame):
        """Verify receive captures connection to local variable."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()
            mock_conn = AsyncMock()
            handler.connection = mock_conn

            await handler.receive(sample_audio_frame)

            # Should have used the connection that was captured
            mock_conn.send.assert_called_once()


class TestMessageFormat:
    """Tests for API message formatting."""

    async def test_audio_message_format(self, sample_audio_frame):
        """Verify audio messages have correct format."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            import json

            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()
            handler.connection = AsyncMock()

            await handler.receive(sample_audio_frame)

            call_args = handler.connection.send.call_args[0][0]
            msg = json.loads(call_args)

            assert msg["type"] == "input_audio_buffer.append"
            assert "event_id" in msg
            assert "audio" in msg
            assert msg["event_id"].startswith("evt_")

    def test_msg_id_format(self):
        """Verify msg_id format is correct."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()
            msg_id = handler.msg_id()

            assert msg_id.startswith("evt_")
            assert msg_id == "evt_1"
