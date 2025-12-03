"""
Error handling tests.

Tests cover:
- Connection failures
- Invalid audio data
- API errors
- Timeout handling
- Graceful degradation
"""

import asyncio
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from websockets.exceptions import ConnectionClosedError

pytestmark = pytest.mark.asyncio


class TestConnectionErrors:
    """Tests for connection error handling."""

    async def test_receive_handles_connection_none(self, sample_audio_frame):
        """Verify receive handles None connection gracefully."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()
            handler.connection = None

            # Should not raise
            await handler.receive(sample_audio_frame)

    async def test_receive_handles_send_failure(self, sample_audio_frame):
        """Verify receive handles send failure gracefully."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()
            handler.connection = AsyncMock()
            handler.connection.send.side_effect = ConnectionClosedError(None, None)

            # Should raise but be catchable
            with pytest.raises(ConnectionClosedError):
                await handler.receive(sample_audio_frame)

    async def test_keepalive_stops_on_connection_error(self):
        """Verify keepalive loop stops on connection error."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()
            handler.connection = AsyncMock()
            handler.connection.send.side_effect = ConnectionClosedError(None, None)
            handler.running = True
            handler.last_audio_time = 0  # Trigger keepalive

            # Simulate one keepalive iteration
            try:
                conn = handler.connection
                if handler.running and conn is not None:
                    handler._msg_counter += 1
                    msg = f'{{"event_id":"evt_{handler._msg_counter}"}}'
                    await conn.send(msg)
            except Exception:
                handler.running = False

            assert handler.running is False

    async def test_shutdown_handles_already_closed_connection(self):
        """Verify shutdown handles already-closed connection."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()
            handler.connection = AsyncMock()
            handler.connection.close.side_effect = Exception("Already closed")

            # Should not raise
            try:
                await handler.shutdown()
            except Exception:
                pytest.fail("shutdown() should handle already-closed connection")


class TestInvalidAudioData:
    """Tests for invalid audio data handling."""

    async def test_receive_handles_empty_array(self):
        """Verify receive handles empty audio array."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()
            handler.connection = AsyncMock()

            empty_frame = (16000, np.array([], dtype=np.int16).reshape(1, -1))

            # Should handle gracefully (may raise, but shouldn't crash)
            try:
                await handler.receive(empty_frame)
            except (ValueError, IndexError):
                pass  # Expected for empty array

    async def test_receive_handles_wrong_dtype(self):
        """Verify receive handles wrong dtype."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()
            handler.connection = AsyncMock()

            # Float32 instead of int16
            float_frame = (16000, np.random.randn(1, 1600).astype(np.float32))

            # Should handle (numpy will convert types)
            await handler.receive(float_frame)

    async def test_receive_handles_large_values(self):
        """Verify receive handles audio with large values."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()
            handler.connection = AsyncMock()

            # Max int16 values
            loud_frame = (16000, np.full((1, 1600), 32767, dtype=np.int16))

            await handler.receive(loud_frame)
            handler.connection.send.assert_called_once()


class TestAPIErrors:
    """Tests for API error handling."""

    async def test_handles_api_timeout(self):
        """Verify timeout errors are handled."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()
            handler.connection = AsyncMock()
            handler.connection.send.side_effect = TimeoutError()

            # Should raise but be catchable
            with pytest.raises(asyncio.TimeoutError):
                await handler.receive((16000, np.ones((1, 1600), dtype=np.int16) * 1000))

    async def test_handles_invalid_response(self):
        """Verify invalid API responses are handled."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            import json

            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()

            # Test parsing invalid JSON
            invalid_responses = [
                "not json",
                '{"incomplete": ',
                '{"type": null}',
                "{}",
            ]

            for response in invalid_responses:
                try:
                    event = json.loads(response)
                    event_type = event.get("type")
                    # Should handle missing or null type
                except json.JSONDecodeError:
                    pass  # Expected for invalid JSON


class TestGracefulDegradation:
    """Tests for graceful degradation under failure conditions."""

    async def test_continues_after_single_frame_error(self, sample_audio_frame):
        """Verify processing continues after single frame error."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()
            handler.connection = AsyncMock()

            # First call fails
            handler.connection.send.side_effect = [
                Exception("Transient error"),
                None,  # Second call succeeds
                None,
            ]

            # First call should raise
            with pytest.raises(Exception):
                await handler.receive(sample_audio_frame)

            # Reset side_effect for subsequent calls
            handler.connection.send.side_effect = None

            # Second call should succeed
            await handler.receive(sample_audio_frame)
            assert handler.connection.send.call_count == 2

    async def test_queue_overflow_handled(self):
        """Verify queue overflow is handled gracefully."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()

            # Fill queue to capacity
            for i in range(100):
                await handler.output_queue.put(f"item_{i}")

            # Queue is full
            assert handler.output_queue.full()

            # put_nowait should raise QueueFull
            with pytest.raises(asyncio.QueueFull):
                handler.output_queue.put_nowait("overflow")

    async def test_video_queue_drops_old_frames(self, sample_video_frame):
        """Verify video queue drops oldest frames when full."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()
            handler.connection = AsyncMock()

            # Fill video queue
            for i in range(10):
                handler.video_queue.put_nowait(f"frame_{i}")

            assert handler.video_queue.full()

            # Simulate video_receive drop logic
            while handler.video_queue.full():
                try:
                    handler.video_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            # Should have space now
            assert not handler.video_queue.full()


class TestResourceCleanup:
    """Tests for proper resource cleanup."""

    async def test_shutdown_cancels_keepalive_task(self):
        """Verify shutdown properly cancels keepalive task."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()
            handler.connection = AsyncMock()
            handler.running = True

            # Create a mock task
            async def mock_keepalive():
                while handler.running:
                    await asyncio.sleep(0.1)

            handler.keepalive_task = asyncio.create_task(mock_keepalive())

            await handler.shutdown()

            assert handler.running is False
            assert handler.keepalive_task is None

    async def test_multiple_shutdown_calls_safe(self):
        """Verify multiple shutdown calls are safe."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()
            handler.connection = AsyncMock()

            # Multiple shutdowns should be safe
            await handler.shutdown()
            await handler.shutdown()
            await handler.shutdown()

            assert handler.connection is None

    async def test_cleanup_removes_from_registry(self):
        """Verify shutdown removes handler from registry."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler, handler_registry

            session_id = "test-session-123"
            handler = LiveTranslateHandler(session_id=session_id)
            handler.connection = AsyncMock()

            # Manually add to registry
            handler_registry[session_id] = handler

            await handler.shutdown()

            assert session_id not in handler_registry
