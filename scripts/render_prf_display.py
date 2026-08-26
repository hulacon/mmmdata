#!/usr/bin/env python3
"""Render the pRF localizer's DISPLAYED movie -- carrier texture through the aperture.

The aperture files are the pRF model's input and contain only the mask. This
renders what the subject actually saw: on each frame a carrier texture is shown
through the aperture against a uniform gray field. Feature-extraction pipelines
(viz2psy and friends) need this, not the mask -- a saliency model run on a bare
black-on-gray aperture measures the aperture, and masking after extraction is
impossible on a naturalistic movie because there is no mask. The whole point of
the calibration is that the pRF stimulus passes through the identical pipeline a
movie would.

PER-RUN, NOT PER-SETNUM. The mask sequence is shared by all 15 runs of a setnum,
but the carrier index is redrawn every run, so a "setnum display movie" would be
a composite nobody saw. This renders one nominated exemplar run per setnum and
records which, so the artifact is a stimulus a real person actually saw.

ORIENTATION: h5py reverses MATLAB v7.3 dimensions, so masks and carriers are
transposed on read (see build_prf_apertures.py, and prf_orientation_check.py for
the empirical confirmation). `specialoverlay` comes from the run mat via
scipy.io, which preserves MATLAB order -- it is NOT transposed. Mixing those two
conventions up is silent and would misalign the overlay against the aperture.

WHAT IS AND IS NOT DRAWN. The faint white fixation grid (`specialoverlay`,
~18% opacity, identical in every run checked) IS composited: it is a static
property of the display. The central fixation dot is NOT: its colour sequence
(`fixationorder`) is a per-run task element, encoding it would need
knkutils-specific reverse engineering, and a high-contrast central dot is a
saliency magnet that would bias exactly the foveal region a saliency->pRF
calibration is trying to measure. `--no-overlay` drops the grid too.

Usage:
    python render_prf_display.py --setnum 93 --subject 03 --session 02 --run 1 \
        --out FILE.mp4 [--resolution 768] [--no-overlay]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))
try:
    from core.config import load_config
    _config = load_config(config_dir=_REPO_ROOT / "config")
    SOURCE_ROOT = Path(_config["paths"]["source_dir"])
except Exception:  # pragma: no cover
    SOURCE_ROOT = Path("/gpfs/projects/hulacon/shared/mmmsourcedata")

WORKSPACE = (SOURCE_ROOT / "shared" / "experiment_code" / "localizer" / "prf"
             / "workspace_retinotopyCaltsmash.mat")
NATIVE = 768
FPS = 15.0


def find_run_mat(subject, session, run, setnum):
    import re
    d = SOURCE_ROOT / f"sub-{subject}" / f"ses-{session}" / "behavioral"
    for p in sorted(d.glob("*_exp9[34].mat")):
        m = re.search(r"_run(\d+)_exp(\d+)\.mat$", p.name)
        if m and int(m.group(1)) == run and int(m.group(2)) == setnum:
            return p
    raise FileNotFoundError(
        f"no run-{run:02d} exp{setnum} mat under {d} -- check --run/--setnum")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setnum", type=int, required=True, choices=(93, 94))
    ap.add_argument("--subject", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resolution", type=int, default=NATIVE)
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--crf", type=int, default=17)
    args = ap.parse_args()

    import h5py
    import scipy.io as sio
    from PIL import Image

    mat_path = find_run_mat(args.subject, args.session, args.run, args.setnum)
    d = sio.loadmat(str(mat_path))
    frameorder = np.asarray(d["frameorder"]).astype(int)
    grayval = int(np.asarray(d["grayval"]).ravel()[0])
    overlay = None if args.no_overlay else np.asarray(d["specialoverlay"])
    carrier_row, mask_row = frameorder[0], frameorder[1]
    n_frames = frameorder.shape[1]
    print(f"exemplar: {mat_path.name}")
    print(f"  {n_frames} frames, gray {grayval}, "
          f"{int((mask_row == 0).sum())} blank")

    used = np.unique(mask_row[mask_row > 0])
    lo, hi = int(used.min()), int(used.max())
    with h5py.File(WORKSPACE, "r") as f:
        print(f"  reading masks {lo}-{hi} ...")
        masks = np.swapaxes(f["maskimages"][lo - 1:hi], 1, 2)   # -> display order
        ref = f["images"][0, 0]
        car = np.asarray(f[ref])                                # (100, 3, H, W)
        carriers = np.swapaxes(car, 2, 3).transpose(0, 2, 3, 1)  # -> (100, H, W, 3)
        print(f"  masks {masks.shape}, carriers {carriers.shape}")

    res = args.resolution
    if overlay is not None:
        oa = (overlay[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
        orgb = overlay[:, :, :3].astype(np.float32)

    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{res}x{res}",
           "-r", f"{FPS}", "-i", "pipe:0",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(args.crf),
           str(args.out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    blank = np.full((NATIVE, NATIVE, 3), grayval, dtype=np.uint8)
    for i in range(n_frames):
        mi, ci = mask_row[i], carrier_row[i]
        if mi == 0:
            frame = blank.astype(np.float32)
        else:
            m = (masks[mi - lo].astype(np.float32) / 255.0)[:, :, None]
            frame = grayval + (carriers[ci - 1].astype(np.float32) - grayval) * m
        if overlay is not None:
            frame = frame * (1.0 - oa) + orgb * oa
        out = np.clip(frame, 0, 255).astype(np.uint8)
        if res != NATIVE:
            out = np.asarray(Image.fromarray(out).resize((res, res), Image.BICUBIC))
        proc.stdin.write(out.tobytes())
        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{n_frames}", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")

    sidecar = Path(str(args.out).replace(".mp4", ".json"))
    with open(sidecar, "w") as fh:
        json.dump({
            "Description": (
                "Displayed pRF localizer stimulus: carrier texture shown through "
                "the aperture on a uniform gray field. This is what the subject "
                "saw, and it is the input a stimulus-feature pipeline should "
                "consume -- not the aperture file, which contains only the mask."
            ),
            "SetNumber": args.setnum,
            "ExemplarRun": {
                "subject": f"sub-{args.subject}", "session": f"ses-{args.session}",
                "run": args.run, "SourceFile": mat_path.name,
                "Note": ("The subject token embedded in the source filename is "
                         "not reliable; the session directory is authoritative. "
                         "Both are recorded deliberately."),
            },
            "WhyPerRun": (
                "The mask sequence is byte-identical across all 15 runs of a "
                "setnum, but the carrier index is redrawn every run. A "
                "per-setnum display movie would therefore be a composite no "
                "subject ever saw; this is one real run."
            ),
            "FrameRate": FPS, "NumberOfFrames": n_frames,
            "Resolution": res, "NativeResolution": NATIVE,
            "BackgroundGrayValue": grayval,
            "FixationGridDrawn": overlay is not None,
            "FixationDotDrawn": False,
            "FixationDotNote": (
                "Deliberately omitted: its colour sequence is a per-run task "
                "element, and a high-contrast central dot would bias any "
                "saliency-derived spatial model toward the fovea."
            ),
            "Orientation": (
                "MATLAB display orientation. Masks and carriers are transposed "
                "on read because h5py reverses MATLAB v7.3 dimensions; "
                "specialoverlay comes from scipy.io and is not."
            ),
            "SourceWorkspace": str(WORKSPACE),
        }, fh, indent=2)
        fh.write("\n")
    print(f"  wrote {args.out} and {sidecar.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
