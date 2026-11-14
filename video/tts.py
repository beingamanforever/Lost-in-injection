"""Synthesize each narration segment with ElevenLabs v4 through OpenRouter's speech endpoint.

Writes video/audio/<id>.mp3 for every segment in narration.json that has no file yet, so editing
one segment and deleting its mp3 re-voices only that segment. Needs OPENROUTER_API_KEY and imageio-ffmpeg.
"""

import json
import os
import subprocess
from pathlib import Path

import imageio_ffmpeg

HERE = Path(__file__).parent
MODEL, VOICE = "elevenlabs/eleven-v4", "alice"
TEMPO = 1.2  # ElevenLabs on OpenRouter ignores `speed`, so the clip is sped up afterwards, pitch kept


def speak(text: str, path: Path) -> None:
    """curl uses the system trust store (needed behind TLS-inspecting proxies); the key goes in on
    stdin, so it never appears in a process list."""
    body = json.dumps({"model": MODEL, "input": text, "voice": VOICE, "response_format": "mp3"})
    subprocess.run(
        ["curl", "-sf", "--max-time", "120", "https://openrouter.ai/api/v1/audio/speech",
         "-H", "@-", "-H", "Content-Type: application/json", "-d", body, "-o", str(path)],
        input=f"Authorization: Bearer {os.environ['OPENROUTER_API_KEY']}", text=True, check=True,
    )


def main() -> None:
    out = HERE / "audio"
    out.mkdir(exist_ok=True)
    for segment in json.loads((HERE / "narration.json").read_text()):
        path = out / f"{segment['id']}.mp3"
        if not path.exists():
            raw = out / f"{segment['id']}.raw.mp3"
            speak(segment["text"], raw)
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-i", str(raw),
                            "-filter:a", f"atempo={TEMPO}", str(path)], check=True)
            raw.unlink()
            print(f"voiced {segment['id']}")


if __name__ == "__main__":
    main()
