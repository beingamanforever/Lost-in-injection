"""Lay a quiet ambient score under the rendered video, ducked whenever the narrator speaks.

The score is synthesized here, so it carries no licence: a slow minor-key pad (A minor, F, C, G),
soft additive voices with a little chorus and a long reverb, a low swell under the title card and
a held final chord. ffmpeg's sidechain compressor lowers it further under each narration clip,
and the mix is normalized to -16 LUFS.
Usage: python video/score.py <manim render.mp4> <out.mp4>
"""

import subprocess
import sys
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np

RATE = 44100
BAR = 5.4  # seconds per chord
# Each chord's notes in Hz: a low root, then a close voicing around middle C.
CHORDS = [
    (55.00, 220.00, 261.63, 329.63, 493.88),  # A minor, add 9
    (43.65, 174.61, 261.63, 329.63, 440.00),  # F major 7
    (65.41, 196.00, 261.63, 329.63, 392.00),  # C major
    (49.00, 196.00, 246.94, 293.66, 392.00),  # G major
]


def voice(freq: float, t: np.ndarray, seed: int) -> np.ndarray:
    """A soft tone: a few decaying harmonics, three detuned copies for width, a slow shimmer."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(t)
    for cents in (-4.0, 0.0, 4.0):
        f = freq * 2 ** (cents / 1200)
        phase = rng.uniform(0, 2 * np.pi)
        for n, gain in ((1, 1.0), (2, 0.28), (3, 0.1), (4, 0.04)):
            out += gain * np.sin(2 * np.pi * f * n * t + n * phase)
    return out * (1 + 0.15 * np.sin(2 * np.pi * rng.uniform(0.05, 0.12) * t))


def score(length: float) -> np.ndarray:
    """Stereo float samples, peak 1."""
    n = int(length * RATE)
    t = np.arange(n) / RATE
    left, right = np.zeros(n), np.zeros(n)
    fade = 1.6  # each chord rises and falls over this many seconds, overlapping the next
    for k in range(int(length // BAR) + 1):
        start = k * BAR - fade / 2
        env = np.clip((t - start) / fade, 0, 1) * np.clip((start + BAR + fade - t) / fade, 0, 1)
        if k == int(length // BAR):  # the last chord holds to the end
            env = np.clip((t - start) / fade, 0, 1)
        env = np.sin(env * np.pi / 2) ** 2
        live = env > 0
        for i, freq in enumerate(CHORDS[k % len(CHORDS)]):
            gain = 0.55 if i == 0 else 0.22
            tone = gain * env[live] * voice(freq, t[live], 100 * k + i)
            pan = 0.5 + 0.18 * np.sin(i * 1.7 + k)  # spread the notes across the stereo field
            left[live] += (1 - pan) * tone
            right[live] += pan * tone
    swell = np.exp(-((t - 1.2) / 0.9) ** 2) * np.sin(2 * np.pi * 41.2 * t)  # under the title
    left, right = left + 0.9 * swell, right + 0.9 * swell
    mix = np.stack([left, right])
    # A long, dark reverb: decaying noise, low-passed, convolved by FFT.
    ir_t = np.arange(int(2.4 * RATE)) / RATE
    ir = np.random.default_rng(1).normal(size=ir_t.size) * np.exp(-ir_t / 0.7)
    ir = np.convolve(ir, np.ones(40) / 40, mode="same")
    size = 1 << int(np.ceil(np.log2(n + ir.size)))
    wet = np.fft.irfft(np.fft.rfft(mix, size) * np.fft.rfft(ir, size), size)[:, :n]
    mix = 0.7 * mix / np.abs(mix).max() + 0.3 * wet / np.abs(wet).max()
    # Gentle low-pass, then fades at both ends.
    kernel = np.hanning(24)
    mix = np.stack([np.convolve(ch, kernel / kernel.sum(), mode="same") for ch in mix])
    edges = np.clip(t / 1.0, 0, 1) * np.clip((length - t) / 2.5, 0, 1)
    mix *= edges
    return mix / np.abs(mix).max()


def main() -> None:
    video, out = Path(sys.argv[1]), Path(sys.argv[2])
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    probe = subprocess.run([ffmpeg, "-i", str(video)], capture_output=True, text=True).stderr
    h, m, s = probe.split("Duration: ")[1].split(",")[0].split(":")
    length = int(h) * 3600 + int(m) * 60 + float(s)
    pcm = (score(length).T * 0.9 * 32767).astype(np.int16)
    music = out.with_suffix(".score.wav")
    with wave.open(str(music), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())
    # The narration keys a compressor on the music, so the score dips by about 10 dB under speech.
    graph = (
        "[0:a]aformat=channel_layouts=stereo,asplit=2[voice][key];"
        "[1:a]volume=0.16[bed];"
        "[bed][key]sidechaincompress=threshold=0.015:ratio=8:attack=30:release=600[ducked];"
        "[voice][ducked]amix=inputs=2:duration=first:normalize=0,"
        "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[a]"  # -16 LUFS, the web standard
    )
    subprocess.run(
        [ffmpeg, "-loglevel", "error", "-y", "-i", str(video), "-i", str(music),
         "-filter_complex", graph, "-map", "0:v", "-map", "[a]",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)],
        check=True,
    )
    music.unlink()


if __name__ == "__main__":
    main()
