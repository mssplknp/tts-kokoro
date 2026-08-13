"""Smoke client: stream one sentence from a running Kokoro TTS server and save
the PCM as a WAV file so you can listen to it.

Usage:
    python scripts/smoke_client.py "नमस्ते किसान भाई।" out.wav
    python scripts/smoke_client.py "Hello world." out.wav --url http://localhost:8001 --rate 44100
"""

import argparse
import wave

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="sentence to synthesize")
    parser.add_argument("out", nargs="?", default="out.wav", help="output wav path")
    parser.add_argument("--url", default="http://localhost:8001", help="server base url")
    parser.add_argument("--rate", type=int, default=None, help="fallback sample rate")
    args = parser.parse_args()

    payload = {"text": args.text}
    chunks: list[bytes] = []

    with httpx.stream(
        "POST",
        f"{args.url}/v1/tts/stream",
        json=payload,
        timeout=None,
    ) as response:
        response.raise_for_status()
        print(
            "response:",
            response.status_code,
            "rate=",
            response.headers.get("X-Sample-Rate"),
            "format=",
            response.headers.get("X-Sample-Format"),
        )
        for chunk in response.iter_bytes():
            if chunk:
                chunks.append(chunk)

    pcm = b"".join(chunks)
    rate = int(response.headers.get("X-Sample-Rate") or args.rate or 44100)

    with wave.open(args.out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)

    print(f"wrote {len(pcm)} bytes of s16le PCM @ {rate} Hz -> {args.out}")


if __name__ == "__main__":
    main()
