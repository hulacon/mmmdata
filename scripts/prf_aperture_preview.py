#!/usr/bin/env python3
"""Render a pRF aperture sequence to a viewable MP4 (or a contact sheet).

The aperture files are raw `(frames, res, res) uint8` arrays -- nothing opens
them by double-clicking. This turns one into an H.264 MP4 that plays anywhere,
at real time by default so what you watch is what the subject saw.

The apertures are the pRF model's input, so this shows the MASK, not the
display: the carrier texture that actually filled the bar is deliberately not
part of the file. A rendered display movie is a different artifact.

Usage:
    python prf_aperture_preview.py --aperture FILE.npy --out FILE.mp4
    python prf_aperture_preview.py --aperture FILE.npy --out FILE.mp4 --speed 10
    python prf_aperture_preview.py --aperture FILE.npy --contact-sheet FILE.png
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

NOMINAL_FPS = 15.0


def write_mp4(seq, out_path, fps, scale):
    """Pipe raw grayscale frames to ffmpeg. Nearest-neighbour upscale keeps the
    aperture edges hard rather than inventing a soft ramp that is not in the
    stimulus."""
    n, h, w = seq.shape
    oh, ow = h * scale, w * scale
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}",
        "-r", f"{fps}", "-i", "pipe:0",
        "-vf", f"scale={ow}:{oh}:flags=neighbor",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        proc.stdin.write(seq[i].tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")


def contact_sheet(seq, out_path, cols=20, rows=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = np.linspace(0, len(seq) - 1, cols * rows).astype(int)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.75, rows * 0.82))
    for ax, f in zip(axes.ravel(), idx):
        ax.imshow(seq[f], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{f / NOMINAL_FPS:.0f}s", fontsize=5, pad=1.5)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, dpi=130)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aperture", required=True, help="path to an *_aperture.npy")
    ap.add_argument("--out", help="output .mp4")
    ap.add_argument("--contact-sheet", help="output .png instead of / as well as mp4")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback multiple of real time (default 1.0 = 15 fps, "
                         "so the full run takes 300 s to watch)")
    ap.add_argument("--scale", type=int, default=4,
                    help="integer upscale for visibility (default 4, so 100 -> 400)")
    args = ap.parse_args()

    seq = np.load(args.aperture)
    if seq.ndim != 3:
        sys.exit(f"expected (frames, res, res), got {seq.shape}")
    print(f"{Path(args.aperture).name}: {seq.shape} {seq.dtype}")

    if args.out:
        fps = NOMINAL_FPS * args.speed
        write_mp4(seq, args.out, fps, args.scale)
        dur = len(seq) / fps
        print(f"  wrote {args.out}  ({fps:.0f} fps, {dur:.0f} s to watch, "
              f"{seq.shape[1] * args.scale}x{seq.shape[2] * args.scale})")
    if args.contact_sheet:
        contact_sheet(seq, args.contact_sheet)
        print(f"  wrote {args.contact_sheet}")
    if not args.out and not args.contact_sheet:
        sys.exit("nothing to do: pass --out and/or --contact-sheet")


if __name__ == "__main__":
    main()
