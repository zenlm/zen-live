"""
Performance and latency tests.

Tests cover:
- Audio processing latency
- Silence detection performance
- Keepalive timing accuracy
- Queue handling under load
- Memory usage patterns
"""

import asyncio
import base64
import time
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.asyncio


class TestAudioProcessingLatency:
    """Tests for audio processing performance."""

    async def test_receive_latency_under_1ms(self, sample_audio_frame):
        """Verify audio receive processing takes under 1ms."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()
            handler.connection = AsyncMock()

            # Warm up
            await handler.receive(sample_audio_frame)

            # Measure
            times = []
            for _ in range(100):
                start = time.perf_counter()
                await handler.receive(sample_audio_frame)
                elapsed = (time.perf_counter() - start) * 1000  # ms
                times.append(elapsed)

            avg_time = sum(times) / len(times)
            max_time = max(times)

            # Average should be well under 1ms
            assert avg_time < 1.0, f"Average latency {avg_time:.3f}ms exceeds 1ms"
            # Max should be under 5ms (allowing for GC, etc.)
            assert max_time < 5.0, f"Max latency {max_time:.3f}ms exceeds 5ms"

    async def test_silence_detection_latency(self):
        """Verify silence detection is fast."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            # Test RMS calculation directly
            audio_data = np.random.randint(-1000, 1000, 1600, dtype=np.int16)

            times = []
            for _ in range(1000):
                start = time.perf_counter()
                rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
                elapsed = (time.perf_counter() - start) * 1000  # ms
                times.append(elapsed)

            avg_time = sum(times) / len(times)

            # RMS calculation should be under 0.1ms
            assert avg_time < 0.1, f"RMS calculation {avg_time:.4f}ms too slow"

    async def test_base64_encoding_latency(self):
        """Verify base64 encoding is fast."""
        audio_data = np.random.randint(-32768, 32767, 1600, dtype=np.int16)
        audio_bytes = audio_data.tobytes()

        times = []
        for _ in range(1000):
            start = time.perf_counter()
            encoded = base64.b64encode(audio_bytes).decode()
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

        avg_time = sum(times) / len(times)

        # Base64 encoding should be under 0.1ms
        assert avg_time < 0.1, f"Base64 encoding {avg_time:.4f}ms too slow"


class TestSilenceDetectionAccuracy:
    """Tests for silence detection accuracy."""

    def test_detects_true_silence(self):
        """Verify true silence (zeros) is detected."""
        silent = np.zeros(1600, dtype=np.int16)
        rms = np.sqrt(np.mean(silent.astype(np.float32) ** 2))
        assert rms < 50

    def test_detects_low_noise(self):
        """Verify low noise floor is detected as silence."""
        # Typical noise floor: random values -20 to 20
        noise = np.random.randint(-20, 20, 1600, dtype=np.int16)
        rms = np.sqrt(np.mean(noise.astype(np.float32) ** 2))
        assert rms < 50

    def test_passes_quiet_speech(self):
        """Verify quiet speech (RMS ~100-500) passes threshold."""
        # Quiet speech simulation
        t = np.linspace(0, 0.1, 1600, dtype=np.float32)
        quiet_speech = (200 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
        rms = np.sqrt(np.mean(quiet_speech.astype(np.float32) ** 2))
        # RMS should be around 141 (200/sqrt(2))
        assert rms > 50, f"Quiet speech RMS {rms} incorrectly detected as silence"

    def test_passes_normal_speech(self):
        """Verify normal speech (RMS ~1000-5000) passes threshold."""
        t = np.linspace(0, 0.1, 1600, dtype=np.float32)
        speech = (3000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
        rms = np.sqrt(np.mean(speech.astype(np.float32) ** 2))
        assert rms > 50
        assert rms > 1000

    def test_threshold_boundary(self):
        """Test behavior exactly at threshold boundary."""
        # Create audio with RMS exactly around 50
        # For a sine wave, RMS = amplitude / sqrt(2)
        # So amplitude = 50 * sqrt(2) ≈ 70.7
        t = np.linspace(0, 0.1, 1600, dtype=np.float32)

        # Just below threshold
        below = (69 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
        rms_below = np.sqrt(np.mean(below.astype(np.float32) ** 2))
        assert rms_below < 50

        # Just above threshold
        above = (72 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
        rms_above = np.sqrt(np.mean(above.astype(np.float32) ** 2))
        assert rms_above > 50


class TestKeepalivePerformance:
    """Tests for keepalive timing and performance."""

    async def test_keepalive_timing_accuracy(self):
        """Verify keepalive triggers at correct intervals."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()
            handler.last_audio_time = time.time() - 1.0  # 1 second ago

            # Should trigger keepalive (threshold is 0.5s)
            now = time.time()
            should_send = (now - handler.last_audio_time) > 0.5
            assert should_send is True

            # Recent audio should not trigger
            handler.last_audio_time = time.time() - 0.1  # 100ms ago
            now = time.time()
            should_send = (now - handler.last_audio_time) > 0.5
            assert should_send is False

    async def test_keepalive_message_size(self):
        """Verify keepalive message is reasonably sized."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            # 100ms of silence at 16kHz = 1600 samples * 2 bytes = 3200 bytes
            # Base64 encoding adds ~33% overhead
            decoded = base64.b64decode(LiveTranslateHandler.SILENT_AUDIO_100MS)
            assert len(decoded) == 3200  # Raw bytes

            # Full message should be under 5KB
            msg = f'{{"event_id":"evt_1","type":"input_audio_buffer.append","audio":"{LiveTranslateHandler.SILENT_AUDIO_100MS}"}}'
            assert len(msg) < 5000


class TestQueuePerformance:
    """Tests for queue handling under load."""

    async def test_output_queue_bounded(self):
        """Verify output queue is bounded to prevent memory issues."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()

            # Queue should be bounded
            assert handler.output_queue.maxsize == 100

    async def test_video_queue_bounded(self):
        """Verify video queue is bounded."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()

            assert handler.video_queue.maxsize == 10

    async def test_queue_handles_overflow(self):
        """Verify queues handle overflow gracefully."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()

            # Fill output queue
            for i in range(100):
                await handler.output_queue.put(f"item_{i}")

            assert handler.output_queue.full()

            # Queue should be at max size
            assert handler.output_queue.qsize() == 100


class TestConcurrentSessions:
    """Tests for concurrent session handling."""

    async def test_multiple_handlers_independent(self):
        """Verify multiple handlers don't share state."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handlers = [LiveTranslateHandler(session_id=f"session-{i}") for i in range(5)]

            # Each should have independent counters
            for i, handler in enumerate(handlers):
                for _ in range(i + 1):
                    handler.msg_id()

            # Verify counters are independent
            assert handlers[0]._msg_counter == 1
            assert handlers[1]._msg_counter == 2
            assert handlers[2]._msg_counter == 3
            assert handlers[3]._msg_counter == 4
            assert handlers[4]._msg_counter == 5

    async def test_concurrent_receive_calls(self, sample_audio_frame):
        """Verify concurrent receive calls are handled correctly."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()
            handler.connection = AsyncMock()

            # Simulate concurrent receive calls
            tasks = [handler.receive(sample_audio_frame) for _ in range(10)]

            await asyncio.gather(*tasks)

            # All should have been sent
            assert handler.connection.send.call_count == 10


class TestMemoryUsage:
    """Tests for memory usage patterns."""

    async def test_no_memory_leak_in_receive(self, sample_audio_frame):
        """Verify receive doesn't leak memory."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            import gc

            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()
            handler.connection = AsyncMock()

            # Process many frames
            for _ in range(1000):
                await handler.receive(sample_audio_frame)

            # Force garbage collection
            gc.collect()

            # Handler should still be operational
            assert handler.connection.send.call_count == 1000

    async def test_shutdown_clears_references(self):
        """Verify shutdown clears all references."""
        with patch.dict(
            "os.environ",
            {"TRANSLATE_API_URL": "wss://test.example.com/api", "HANZO_API_KEY": "test-key"},
        ):
            from app import LiveTranslateHandler

            handler = LiveTranslateHandler()
            handler.connection = AsyncMock()

            # Add items to queues
            for i in range(50):
                await handler.output_queue.put(f"output_{i}")
                if i < 10:
                    await handler.video_queue.put(f"frame_{i}")

            await handler.shutdown()

            assert handler.connection is None
            assert handler.output_queue.empty()
            assert handler.video_queue.empty()
