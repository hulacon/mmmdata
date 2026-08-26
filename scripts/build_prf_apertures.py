#!/usr/bin/env python3
"""
Reconstruct the pRF localizer aperture sequences from the experiment workspace.

The pRF localizer was run from knkutils `showmulticlass` against the same
aperture stack that Kendrick Kay's analyzePRF ships as its worked example
(768 x 768 x 2580 uint8 masks, 15 Hz, 4500 frames = 300 s). A displayed frame
is carrier[frameorder[0]] shown through mask[frameorder[1]]; the pRF model
consumes the *mask* only -- carrier content never enters it.

Two facts make the aperture a per-setnum artifact rather than a per-run one:

  frameorder row 0  carrier image index, 1-100     differs in every run
  frameorder row 1  mask index into `maskimages`   byte-identical across all
                                                   runs of a setnum

so one file per setnum represents every run of that setnum exactly. `validate`
asserts that identity rather than assuming it, and audits frame timing while it
has the run mats open.

Both MAT readers are required and neither substitutes for the other: the
experiment workspace is v7.3 (HDF5 -> h5py) and the per-run mats are pre-v7.3
(-> scipy.io.loadmat).

ORIENTATION, the trap that matters most. h5py returns a MATLAB v7.3 array with
its dimensions REVERSED, so the 768x768 frame it hands back is the TRANSPOSE of
the one MATLAB drew to the screen. Every frame is therefore transposed on read.
This is not a cosmetic detail: without it polar angle is mirrored, V1/V2/V3
borders land in the wrong place, and the resulting maps still look like
textbook retinotopy. Confirmed empirically against anatomy, not just reasoned
about -- see `prf_orientation_check.py`, which uses the fact that each
hemisphere represents the contralateral field (r = -0.67 between preferred
horizontal position and MNI x) and that dorsal V1 represents the lower field
(r = -0.28 vs MNI z).

Index conventions, easy to get wrong:
  - `frameorder` is MATLAB 1-based; 0 means a blank (uniform gray) frame, and
    row 0 and row 1 are zero on exactly the same frames.
  - `maskimages` is read 0-based here, so mask index i maps to maskimages[i-1].
  - Mask values are 0-255, "0 = do not pass the pattern, 255 = fully pass".
    Written out as uint8; a consumer divides by 255.

Usage:
    python build_prf_apertures.py validate
    python build_prf_apertures.py build --out-dir DIR [--resolution 100]
    python build_prf_apertures.py build --out-dir DIR --resolution 200

Writing into the BIDS `stimuli/` tree may be blocked depending on how this is
invoked; `--out-dir` exists so the artifacts can be staged and moved by hand.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup -- use config if importable, else fall back to well-known path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent

sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))
try:
    from core.config import load_config
    _config = load_config(config_dir=_REPO_ROOT / "config")
    BIDS_ROOT = Path(_config["paths"]["bids_project_dir"])
    SOURCE_ROOT = Path(_config["paths"]["source_dir"])
except Exception:  # pragma: no cover - config is present in every deployment
    BIDS_ROOT = Path("/gpfs/projects/hulacon/shared/mmmdata")
    SOURCE_ROOT = Path("/gpfs/projects/hulacon/shared/mmmsourcedata")

WORKSPACE = (SOURCE_ROOT / "shared" / "experiment_code" / "localizer" / "prf"
             / "workspace_retinotopyCaltsmash.mat")

# Run mats are named <YYYYMMDDHHMMSS>_subj<N>_run<NN>_exp<SS>.mat. The subject
# token in that name is NOT reliable (one session's console numbering disagrees
# with its directory); pair mats to runs by session directory and run index.
RUN_MAT_GLOB = "sub-*/ses-*/behavioral/*_exp9[34].mat"
RUN_MAT_RE = re.compile(r"_run(?P<run>\d+)_exp(?P<setnum>\d+)\.mat$")

N_FRAMES = 4500           # stimulus frames per run, 15 Hz
FRAME_HZ = 15.0
N_MASKS = 2580            # frames in the shared aperture stack
DEFAULT_RESOLUTION = 100  # analyzePRF's shipped example downsamples 768 -> 100


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_run_mats():
    """Return [(subject, session, run, setnum, path), ...] sorted, for all runs."""
    rows = []
    for path in sorted(SOURCE_ROOT.glob(RUN_MAT_GLOB)):
        m = RUN_MAT_RE.search(path.name)
        if not m:
            raise ValueError(f"run mat name does not parse: {path}")
        # .../sub-##/ses-##/behavioral/<file>.mat
        session = path.parent.parent.name
        subject = path.parent.parent.parent.name
        rows.append((subject, session, int(m.group("run")),
                     int(m.group("setnum")), path))
    if not rows:
        raise FileNotFoundError(
            f"no pRF run mats under {SOURCE_ROOT}/{RUN_MAT_GLOB} -- "
            f"check paths.source_dir in config"
        )
    return rows


def read_run_mat(path):
    """Read one pre-v7.3 run mat. Returns (frameorder, timeframes, setnum)."""
    import scipy.io as sio
    d = sio.loadmat(str(path))
    for key in ("frameorder", "timeframes", "setnum"):
        if key not in d:
            raise KeyError(f"{path.name}: missing '{key}'")
    return (np.asarray(d["frameorder"]),
            np.asarray(d["timeframes"]).ravel(),
            int(np.asarray(d["setnum"]).ravel()[0]))


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def cmd_validate(args):
    runs = find_run_mats()
    by_setnum = defaultdict(list)
    clocks = defaultdict(list)
    problems = []
    print(f"{len(runs)} pRF run mats under {SOURCE_ROOT}\n")
    header = (f"{'subject':>8} {'session':>8} {'run':>4} {'set':>4} "
              f"{'frames':>7} {'span_s':>8} {'med_dt_ms':>10} {'max_dt_ms':>10} "
              f"{'drift_ms':>9} {'blank':>6} {'mask_row_sha1':>14}")
    print(header)
    print("-" * len(header))

    for subject, session, run, setnum_name, path in runs:
        frameorder, timeframes, setnum = read_run_mat(path)

        if setnum != setnum_name:
            problems.append(f"{path.name}: filename says exp{setnum_name}, "
                            f"mat says setnum {setnum}")
        if frameorder.shape != (2, N_FRAMES):
            problems.append(f"{path.name}: frameorder {frameorder.shape}, "
                            f"expected (2, {N_FRAMES})")
            continue

        carrier_row, mask_row = frameorder[0], frameorder[1]
        if not np.array_equal(carrier_row == 0, mask_row == 0):
            problems.append(f"{path.name}: blank frames disagree between "
                            f"carrier and mask rows")

        mask_sha1 = hashlib.sha1(np.ascontiguousarray(
            mask_row.astype(np.uint16)).tobytes()).hexdigest()
        by_setnum[setnum].append((path.name, mask_sha1, mask_row))

        dt = np.diff(timeframes)
        span = float(timeframes[-1] - timeframes[0])
        max_dt_ms = float(dt.max() * 1000.0) if dt.size else float("nan")
        med_dt_ms = float(np.median(dt) * 1000.0) if dt.size else float("nan")
        # Cumulative slip of the display clock against the nominal 15 Hz over
        # the whole run. Positive = the stimulus ran long.
        drift_ms = (span - dt.size / FRAME_HZ) * 1000.0
        clocks[round(1000.0 / med_dt_ms, 2)].append(
            f"{subject}/{session}/run-{run:02d}")
        n_blank = int((mask_row == 0).sum())

        # A dropped frame shows as an inter-frame interval near a multiple of
        # the nominal 66.7 ms. Flag anything past 1.5 nominal frames.
        if dt.size and dt.max() > 1.5 / FRAME_HZ:
            problems.append(f"{path.name}: max inter-frame interval "
                            f"{max_dt_ms:.1f} ms -- frame(s) dropped")

        print(f"{subject:>8} {session:>8} {run:>4} {setnum:>4} "
              f"{frameorder.shape[1]:>7} {span:>8.3f} {med_dt_ms:>10.3f} "
              f"{max_dt_ms:>10.1f} {drift_ms:>9.1f} "
              f"{n_blank:>6} {mask_sha1[:14]:>14}")

    print()
    for rate in sorted(clocks):
        members = clocks[rate]
        subjects = sorted({m.split("/")[0] for m in members})
        print(f"frame clock {rate:.2f} Hz (display {rate * 4:.2f} Hz): "
              f"{len(members)} runs, {', '.join(subjects)}")
    if len(clocks) > 1:
        print("  NOTE: more than one display clock is present. The frame "
              "sequence is unaffected -- it is the same masks in the same "
              "order -- but the frame-to-TR mapping is not, and is a per-run "
              "property. See the sidecar's FrameClockRegimes.")
    print()
    for setnum in sorted(by_setnum):
        entries = by_setnum[setnum]
        hashes = {h for _, h, _ in entries}
        mask_row = entries[0][2]
        nz = mask_row[mask_row > 0]
        status = "IDENTICAL" if len(hashes) == 1 else "*** DIFFER ***"
        print(f"setnum {setnum}: {len(entries)} runs, mask row {status} "
              f"({len(hashes)} distinct sha1); nonzero mask index range "
              f"{nz.min()}-{nz.max()}, {len(np.unique(nz))} distinct masks, "
              f"{int((mask_row == 0).sum())} blank frames")
        if len(hashes) != 1:
            problems.append(f"setnum {setnum}: mask row is not identical "
                            f"across its {len(entries)} runs")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nOK -- every run's mask row matches its setnum; no dropped frames.")
    return 0


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def load_mask_stack(resolution, chunk=64):
    """Read `maskimages` from the v7.3 workspace, downsampled to resolution.

    Returned as (N_MASKS, resolution, resolution) uint8, 0-based: mask index i
    from frameorder is stack[i - 1]. Read in chunks so peak memory stays near
    the output size rather than the 1.5 GB source array.
    """
    import h5py
    from PIL import Image

    if not WORKSPACE.exists():
        raise FileNotFoundError(f"experiment workspace not found: {WORKSPACE}")

    out = np.empty((N_MASKS, resolution, resolution), dtype=np.uint8)
    with h5py.File(WORKSPACE, "r") as f:
        masks = f["maskimages"]
        if masks.shape[0] != N_MASKS:
            raise ValueError(f"maskimages has {masks.shape[0]} frames, "
                             f"expected {N_MASKS}")
        native = masks.shape[1]
        for start in range(0, N_MASKS, chunk):
            stop = min(start + chunk, N_MASKS)
            block = masks[start:stop]
            # Transpose back into MATLAB's display orientation -- see the
            # ORIENTATION note in the module docstring. Done before resizing so
            # the two operations cannot be confused with one another.
            block = np.swapaxes(block, 1, 2)
            if resolution == native:
                out[start:stop] = block
            else:
                for k in range(stop - start):
                    # Bicubic with PIL's support scaling is the antialiased
                    # downsample MATLAB's imresize(...,'cubic') performs, which
                    # is what analyzePRF's example applies to this same stack.
                    img = Image.fromarray(np.ascontiguousarray(block[k])).resize(
                        (resolution, resolution), Image.BICUBIC)
                    out[start + k] = np.asarray(img, dtype=np.uint8)
            print(f"  masks {stop}/{N_MASKS}", end="\r", flush=True)
    print(f"  masks {N_MASKS}/{N_MASKS} -> {resolution}x{resolution}")
    return out


def per_tr_average(sequence, n_tr, n_frames=N_FRAMES):
    """Average the frame-resolution aperture into one frame per TR.

    4500 frames / 200 TRs = 22.5, so TR boundaries fall mid-frame. Frames are
    weighted by their overlap with the TR window rather than assigned to the
    nearest TR, which would alternate 22- and 23-frame bins and put a spurious
    half-frame jitter into the regressor.
    """
    seq = sequence.astype(np.float32)
    edges = np.linspace(0.0, n_frames, n_tr + 1)
    out = np.empty((n_tr, seq.shape[1], seq.shape[2]), dtype=np.float32)
    for t in range(n_tr):
        lo, hi = edges[t], edges[t + 1]
        first, last = int(np.floor(lo)), int(np.ceil(hi))
        idx = np.arange(first, last)
        w = np.minimum(idx + 1.0, hi) - np.maximum(idx.astype(np.float64), lo)
        w = w / w.sum()
        out[t] = np.tensordot(w.astype(np.float32), seq[first:last], axes=(0, 0))
    return out


def cmd_build(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = find_run_mats()
    canonical = {}   # setnum -> (mask_row, [provenance run labels])
    clocks = defaultdict(list)
    for subject, session, run, _setnum_name, path in runs:
        frameorder, timeframes, setnum = read_run_mat(path)
        mask_row = np.asarray(frameorder[1]).astype(np.int32)
        if setnum not in canonical:
            canonical[setnum] = (mask_row, [])
        elif not np.array_equal(canonical[setnum][0], mask_row):
            raise ValueError(
                f"setnum {setnum}: {path.name} disagrees with the canonical "
                f"mask row -- run `validate` before building"
            )
        label = f"{subject}/{session}/run-{run:02d}"
        canonical[setnum][1].append(label)
        rate = round(1.0 / float(np.median(np.diff(timeframes))), 2)
        clocks[rate].append(label)

    regimes = [
        {
            "FrameRateHz": rate,
            "ImpliedDisplayRefreshHz": round(rate * 4, 2),
            "DriftVsNominalMsPerRun": round(
                (N_FRAMES - 1) * (1.0 / rate - 1.0 / FRAME_HZ) * 1000.0, 1),
            "Runs": sorted(clocks[rate]),
        }
        for rate in sorted(clocks)
    ]

    print(f"validated {len(runs)} runs into {len(canonical)} setnums, "
          f"{len(regimes)} display-clock regime(s)")
    stack = load_mask_stack(args.resolution)

    for setnum in sorted(canonical):
        mask_row, provenance = canonical[setnum]
        blank = mask_row == 0
        seq = np.zeros((N_FRAMES, args.resolution, args.resolution),
                       dtype=np.uint8)
        seq[~blank] = stack[mask_row[~blank] - 1]

        stem = f"task-prf_set-{setnum}_res-{args.resolution}"
        seq_path = out_dir / f"{stem}_aperture.npy"
        np.save(seq_path, seq)

        nz = mask_row[mask_row > 0]
        sidecar = {
            "Description": (
                "pRF localizer aperture sequence in presentation order, "
                "reconstructed from the experiment workspace. Values 0-255; "
                "0 = do not pass the pattern, 255 = fully pass. Divide by 255 "
                "for a 0-1 stimulus. The carrier texture is deliberately "
                "absent: it never enters the pRF model."
            ),
            "SetNumber": setnum,
            "Shape": list(seq.shape),
            "DType": "uint8",
            "NominalFrameRate": FRAME_HZ,
            "NumberOfFrames": N_FRAMES,
            "BlankFrames": int(blank.sum()),
            "MaskIndexRange": [int(nz.min()), int(nz.max())],
            "DistinctMasks": int(np.unique(nz).size),
            "NativeResolution": 768,
            "Resolution": args.resolution,
            "Orientation": (
                "MATLAB display orientation: row index = screen vertical "
                "(row 0 = top), column index = screen horizontal (column 0 = "
                "left). The source workspace is read with h5py, which reverses "
                "MATLAB v7.3 dimensions, so each frame is transposed on read. "
                "Verified against anatomy (contralateral hemifields r=-0.67 vs "
                "MNI x; dorsal V1 = lower field r=-0.28 vs MNI z), not assumed."
            ),
            "ResampleMethod": (
                "PIL bicubic with support scaling (antialiased), "
                "matching imresize(...,'cubic')"
            ),
            "SourceWorkspace": str(WORKSPACE),
            "RepresentsRuns": sorted(provenance),
            "ValidationNote": (
                "The mask-index row of frameorder is byte-identical across "
                "every run of this setnum; this file therefore represents all "
                "of them exactly. The carrier-index row differs per run and is "
                "not reconstructed here."
            ),
            "FrameClockRegimes": regimes,
            "FrameToTRMappingNote": (
                "This file is a frame sequence, not a TR-resolution design "
                "matrix, and deliberately so. Mapping frames onto TRs needs "
                "two per-run facts that are not properties of the stimulus: "
                "the run's actual frame clock (see FrameClockRegimes -- the "
                "display refresh is not the same for every subject) and the "
                "offset between the first stimulus frame and the first "
                "retained volume. Build the TR-resolution stimulus per run "
                "from that run's `timeframes`, and verify the offset before "
                "trusting any fit: a constant frame/TR offset yields maps that "
                "look like textbook retinotopy while being wrong."
            ),
        }

        if args.per_tr:
            per_tr = per_tr_average(seq, args.n_tr)
            tr_path = out_dir / f"{stem}_desc-perTRnominal_aperture.npy"
            np.save(tr_path, per_tr.astype(np.float32))
            sidecar["PerTRCompanion"] = {
                "File": tr_path.name,
                "Shape": [args.n_tr, args.resolution, args.resolution],
                "DType": "float32",
                "Description": (
                    f"Frame sequence averaged into {args.n_tr} TRs on the "
                    f"NOMINAL {FRAME_HZ} Hz clock, assuming zero offset "
                    f"between the first frame and the first volume. Both "
                    f"assumptions are wrong for at least one clock regime -- "
                    f"see FrameToTRMappingNote. Convenience only; do not fit "
                    f"from this without checking it against the run."
                ),
            }
            print(f"  set-{setnum}: {tr_path.name} "
                  f"{per_tr.nbytes / 2**20:.1f} MiB (nominal clock)")

        with open(out_dir / f"{stem}_aperture.json", "w") as fh:
            json.dump(sidecar, fh, indent=2)
            fh.write("\n")

        print(f"  set-{setnum}: {seq_path.name} "
              f"{seq.nbytes / 2**20:.1f} MiB, {len(provenance)} runs represented")

    print(f"\nwrote to {out_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="audit all run mats; exit 1 on any "
                                            "mismatch or dropped frame")
    p_val.set_defaults(func=cmd_validate)

    p_bld = sub.add_parser("build", help="write one aperture sequence per setnum")
    p_bld.add_argument("--out-dir", required=True,
                       help="destination directory (created if absent)")
    p_bld.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION,
                       help=f"square output resolution (default "
                            f"{DEFAULT_RESOLUTION}; analyzePRF's example uses "
                            f"100, 200 where foveal accuracy matters, 768 is "
                            f"native and large)")
    p_bld.add_argument("--per-tr", action="store_true",
                       help="also write a TR-resolution companion on the "
                            "nominal clock. Off by default: it assumes a "
                            "frame rate and a trigger offset that are per-run "
                            "properties, and one of them is unverified")
    p_bld.add_argument("--n-tr", type=int, default=200,
                       help="TRs per run, for --per-tr (default 200)")
    p_bld.set_defaults(func=cmd_build)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
