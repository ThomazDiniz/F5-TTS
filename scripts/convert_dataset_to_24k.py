"""
One-off script: convert all WAVs in a dataset folder to target sample rate, 16-bit, mono.
Usage: python scripts/convert_dataset_to_24k.py "path/to/dataset" [--sr 16000]
"""
import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def convert_file(path: Path, target_sr: int) -> None:
    audio, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    # 16-bit: scale to [-32768, 32767]
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    sf.write(path, audio_int16, target_sr, subtype="PCM_16")


def main():
    parser = argparse.ArgumentParser(description="Convert dataset WAVs to target sample rate, 16-bit, mono")
    parser.add_argument("dir", type=str, help="Dataset directory (e.g. .../dataset)")
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate (default: 16000)")
    args = parser.parse_args()
    dir_path = Path(args.dir)
    if not dir_path.is_dir():
        print(f"Error: not a directory: {dir_path}", file=sys.stderr)
        sys.exit(1)
    wavs = list(dir_path.glob("*.wav"))
    if not wavs:
        print(f"No .wav files in {dir_path}", file=sys.stderr)
        sys.exit(1)
    for i, p in enumerate(wavs):
        try:
            convert_file(p, args.sr)
            print(f"[{i+1}/{len(wavs)}] {p.name}")
        except Exception as e:
            print(f"Error converting {p}: {e}", file=sys.stderr)
    print(f"Done. Converted {len(wavs)} files to {args.sr} Hz, 16-bit, mono.")


if __name__ == "__main__":
    main()
