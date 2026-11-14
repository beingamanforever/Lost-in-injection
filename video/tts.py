"""Synthesize each narration segment with ElevenLabs v4 through OpenRouter's speech endpoint.

Writes video/audio/<id>.mp3 for every segment in narration.json that has no file yet, so editing
one segment and deleting its mp3 re-voices only that segment. Needs OPENROUTER_API_KEY.
"""

import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
MODEL, VOICE = "elevenlabs/eleven-v4", "george"


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
            speak(segment["text"], path)
            print(f"voiced {segment['id']}")


if __name__ == "__main__":
    main()
