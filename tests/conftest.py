"""
Pytest configuration and shared fixtures for Zen Live tests.
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_audio_frame():
    """Generate a sample audio frame with actual sound (sine wave)."""
    # 16kHz sample rate, 100ms of audio = 1600 samples
    sample_rate = 16000
    duration = 0.1  # 100ms
    samples = int(sample_rate * duration)

    # Generate 440Hz sine wave (audible tone)
    t = np.linspace(0, duration, samples, dtype=np.float32)
    amplitude = 5000  # Reasonable amplitude for int16
    audio = (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.int16)

    return (sample_rate, audio.reshape(1, -1))


@pytest.fixture
def silent_audio_frame():
    """Generate a silent audio frame (all zeros)."""
    sample_rate = 16000
    samples = 1600  # 100ms
    audio = np.zeros(samples, dtype=np.int16)
    return (sample_rate, audio.reshape(1, -1))


@pytest.fixture
def quiet_audio_frame():
    """Generate a very quiet audio frame (RMS ~30, below threshold)."""
    sample_rate = 16000
    samples = 1600
    # Small random noise with RMS around 30
    audio = np.random.randint(-40, 40, samples, dtype=np.int16)
    return (sample_rate, audio.reshape(1, -1))


@pytest.fixture
def loud_audio_frame():
    """Generate a loud audio frame (RMS ~10000)."""
    sample_rate = 16000
    samples = 1600
    t = np.linspace(0, 0.1, samples, dtype=np.float32)
    amplitude = 15000
    audio = (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    return (sample_rate, audio.reshape(1, -1))


@pytest.fixture
def sample_video_frame():
    """Generate a sample video frame."""
    # 640x480 RGB frame
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    ws = AsyncMock()
    ws.send = AsyncMock()
    ws.recv = AsyncMock()
    ws.close = AsyncMock()
    return ws


@pytest.fixture
def mock_session_config():
    """Default session configuration for tests."""
    return {"src_language": "Spanish", "target_language": "English", "voice": "Cherry"}


@pytest.fixture
def env_vars():
    """Set up required environment variables for testing."""
    original_env = os.environ.copy()
    os.environ.update(
        {
            "TRANSLATE_API_URL": "wss://test.example.com/api",
            "HANZO_API_KEY": "test-api-key",
            "BASE_URL": "https://test.zen-live.hanzo.ai",
        }
    )
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def audio_rms_calculator():
    """Helper function to calculate RMS of audio data."""

    def calculate_rms(audio_data: np.ndarray) -> float:
        return float(np.sqrt(np.mean(audio_data.astype(np.float32) ** 2)))

    return calculate_rms
