#!/usr/bin/env python3
"""Quick test: load a .wav and call transcribe() (Whisper may download on first use).

From repo root with local Python (PYTHONPATH=src):
  python scripts/sanity_transcribe_wav.py data/all_filtered_char/wavs/segment_0.wav

Same code path as Gradio, inside Docker:
  run_docker_transcribe_test.bat
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(repo, "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    wav = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        repo, "data", "all_filtered_char", "wavs", "segment_0.wav"
    )
    wav = os.path.abspath(os.path.normpath(wav))
    if not os.path.isfile(wav):
        print(f"ERRO: ficheiro nao existe: {wav}", file=sys.stderr)
        return 1

    import f5_tts.infer.utils_infer as ui

    print(f"utils_infer: {ui.__file__}", flush=True)
    print(f"wav: {wav}", flush=True)

    from f5_tts.infer.utils_infer import transcribe

    text = transcribe(wav, "portuguese")
    print(f"OK texto ({len(text)} chars): {text[:500]!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
