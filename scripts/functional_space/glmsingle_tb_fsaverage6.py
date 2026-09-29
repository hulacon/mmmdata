#!/usr/bin/env python3
"""
glmsingle_tb_fsaverage6.py — the `enc` arm of glmsingle_tb.py, refitted on
fsaverage6 cortical vertices.

Same runs, events, condition mapping, design matrices, TR, stimdur and
GLMsingle options as ``scripts/glmsingle_tb.py --arm enc``: those are imported
from that module, not copied, so the two fits cannot drift apart. Only the
data change:

  - fMRIPrep ``hemi-{L,R}_space-fsaverage6_bold.func.gii`` per run,
    left hemisphere then right, concatenated along vertices;
  - restricted to FreeSurfer's fsaverage6 ``?h.cortex.label`` (the medial wall
    is dropped). This is the mask the volume fit never had: that fit ran over
    every voxel of the 4-D grid, air included.

GLMsingle takes the result as 2-D (vertices x time) data. Row i of every
output array is the vertex named by row i of ``vertex_index.tsv``.

Outputs go to ``<derivatives>/functional_space/glmsingle_tb_fsaverage6/
sub-##/enc/``, laid out like ``glmsingle_tb/sub-##/enc/``. The volume tree's
directories carry no space entity, so a surface fit beside the volume fit
would be told apart only by array shape; a separate tree keeps the two
unambiguous.

Before fitting, the run list, condition key and trial table are checked
against the volume fit's own ``run_metadata.json`` / ``condition_key.csv`` /
``trial_info.csv``; any mismatch is an error, so the surface betas line up
trial-for-trial with the volume betas.

Usage:
    python glmsingle_tb_fsaverage6.py --subject sub-## --dry-run
    python glmsingle_tb_fsaverage6.py --subject sub-##
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPTS))
import glmsingle_tb as base  # noqa: E402  (the volume driver; design source)

ARM = "enc"
SPACE = "fsaverage6"
HEMIS = ("L", "R")
N_VERTICES_HEMI = 40962
TREE = Path("functional_space") / "glmsingle_tb_fsaverage6"

# The GLMsingle options of base.run(), restated so the sidecar can record
# them; assert_same_params() fails if base.run() ever changes them.
GLMSINGLE_PARAMS = {
    "wantlibrary": 1, "wantglmdenoise": 1, "wantfracridge": 1,
    "wantfileoutputs": [1, 1, 1, 1], "wantmemoryoutputs": [0, 0, 0, 0],
}


def assert_same_params():
    """The volume driver builds its params dict inline; check its source still
    says what GLMSINGLE_PARAMS says, so the two fits share every option."""
    import inspect
    src = inspect.getsource(base.run)
    for k, v in GLMSINGLE_PARAMS.items():
        if f'"{k}": {v}' not in src:
            sys.exit(f"ERROR: glmsingle_tb.run() no longer sets {k}={v}; "
                     "update GLMSINGLE_PARAMS to match before fitting.")
    if "extra_regressors" in GLMSINGLE_PARAMS:
        sys.exit("ERROR: unexpected extra_regressors")


def git_sha(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def git_dirty(path, relpath):
    try:
        out = subprocess.run(["git", "-C", str(path), "status", "--porcelain", relpath],
                             capture_output=True, text=True, check=True).stdout.strip()
        return bool(out)
    except Exception:
        return None


# ── surface data ─────────────────────────────────────────────────────────────

def gii_path(fmriprep_dir, subject, session, task, run, hemi):
    return (fmriprep_dir / subject / session / "func"
            / f"{subject}_{session}_task-{task}_{run}_hemi-{hemi}_space-{SPACE}_bold.func.gii")


def cortex_mask(fmriprep_dir):
    """Boolean mask over the L+R fsaverage6 vertices: True on cortex.

    Read from the fsaverage6 subject fMRIPrep sampled onto, so the mask and
    the data share one mesh."""
    label_dir = fmriprep_dir / "sourcedata" / "freesurfer" / "fsaverage6" / "label"
    parts = []
    for hemi in HEMIS:
        lab = label_dir / f"{hemi.lower()}h.cortex.label"
        if not lab.exists():
            sys.exit(f"ERROR: cortex label not found: {lab}")
        m = np.zeros(N_VERTICES_HEMI, dtype=bool)
        m[nib.freesurfer.read_label(str(lab))] = True
        parts.append(m)
    return np.concatenate(parts)


def vertex_index(mask):
    hemi = np.repeat(np.array(HEMIS), N_VERTICES_HEMI)
    vert = np.tile(np.arange(N_VERTICES_HEMI), len(HEMIS))
    return pd.DataFrame({"row": np.arange(int(mask.sum())),
                         "hemi": hemi[mask], "vertex": vert[mask]})


def load_run(fmriprep_dir, subject, session, task, run, mask):
    """(n_cortex_vertices, n_volumes) float32 for one run."""
    halves = []
    for hemi in HEMIS:
        g = nib.load(str(gii_path(fmriprep_dir, subject, session, task, run, hemi)))
        arr = np.stack([d.data for d in g.darrays], axis=1).astype(np.float32)
        if arr.shape[0] != N_VERTICES_HEMI:
            sys.exit(f"ERROR: {session}/{run} hemi-{hemi} has {arr.shape[0]} vertices")
        halves.append(arr)
    return np.concatenate(halves, axis=0)[mask]


# ── agreement with the volume fit ────────────────────────────────────────────

def check_against_volume(vol_dir, run_list, condition_key, trial_info):
    """The surface fit must reproduce the volume fit's design exactly."""
    meta = json.loads((vol_dir / "run_metadata.json").read_text())
    ours = [f"{s}/{r}[enc]" for s, _, r in run_list]
    if meta["run_labels"] != ours:
        sys.exit("ERROR: run list differs from the volume fit's run_metadata.json")

    vk = pd.read_csv(vol_dir / "condition_key.csv", dtype={"condition_id": str, "mmmId": str})
    ok = condition_key[["col_index", "condition_id", "n_presentations"]].reset_index(drop=True)
    vk = vk[["col_index", "condition_id", "n_presentations"]].reset_index(drop=True)
    vk["condition_id"] = vk["condition_id"].map(base.norm_mmm)
    if not ok.astype(str).equals(vk.astype(str)):
        sys.exit("ERROR: condition_key differs from the volume fit's")

    vt = pd.read_csv(vol_dir / "trial_info.csv")
    cols = ["session", "run", "run_idx", "onset", "col_index"]
    a = trial_info[cols].reset_index(drop=True)
    b = vt[cols].reset_index(drop=True)
    if len(a) != len(b) or not (a.astype(str).values == b.astype(str).values).all():
        sys.exit("ERROR: trial_info differs from the volume fit's (order or content)")
    print(f"  Matches the volume fit: {len(ours)} runs, {len(ok)} conditions, "
          f"{len(a)} trials in the same order")


# ── runner ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", required=True, help="e.g. sub-##")
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve inputs, build and check the design, fit nothing")
    args = p.parse_args()

    assert_same_params()
    cfg = base.load_config()
    bids_root = Path(cfg["bids_project_dir"])
    deriv = Path(cfg.get("output_dir", bids_root / "derivatives"))
    fmriprep_dir = deriv / "fmriprep"
    vol_dir = deriv / base.OUTPUT_TREE / args.subject / ARM
    out_base = deriv / TREE
    out_dir = out_base / args.subject / ARM
    print(f"fMRIPrep:    {fmriprep_dir}\nVolume fit:  {vol_dir}\nOutput:      {out_dir}")

    session_runs = base.discover_sessions(bids_root, fmriprep_dir, args.subject, ARM)
    run_list = [(s, t, r) for s, typed in session_runs for t, r, _ in typed]
    session_indices = [i + 1 for i, (s, typed) in enumerate(session_runs) for _ in typed]
    run_labels = [f"{s}/{r}[enc]" for s, _, r in run_list]

    missing = [str(gii_path(fmriprep_dir, args.subject, s, t, r, h))
               for s, t, r in run_list for h in HEMIS
               if not gii_path(fmriprep_dir, args.subject, s, t, r, h).exists()]
    if missing:
        sys.exit(f"ERROR: {len(missing)} surface files missing, e.g. {missing[0]}")

    all_events = base.load_all_events(bids_root, args.subject, session_runs)
    cond_map, condition_key = base.build_condition_mapping(all_events, ARM)
    n_vols = [len(nib.load(str(gii_path(fmriprep_dir, args.subject, s, t, r, "L"))).darrays)
              for s, t, r in run_list]
    designs, trial_info = base.build_design_matrices(all_events, cond_map, n_vols,
                                                     run_labels, ARM)
    check_against_volume(vol_dir, run_list, condition_key, trial_info)

    mask = cortex_mask(fmriprep_dir)
    vidx = vertex_index(mask)
    print(f"  Cortex vertices: {int(mask.sum())} of {mask.size} "
          f"(L {int(mask[:N_VERTICES_HEMI].sum())}, R {int(mask[N_VERTICES_HEMI:].sum())})")

    sha = git_sha(_SCRIPTS.parent)
    sidecar = {
        "Description": "GLMsingle single-trial betas, TB encoding, fsaverage6 cortex. "
                       "Same design as the glmsingle_tb enc arm; only the data differ.",
        "subject": args.subject, "arm": ARM, "space": SPACE,
        "data": "fMRIPrep hemi-L then hemi-R space-fsaverage6 bold.func.gii, "
                "rows restricted to fsaverage6 ?h.cortex.label (medial wall dropped)",
        "mask_source": str(fmriprep_dir / "sourcedata/freesurfer/fsaverage6/label"),
        "n_vertices": int(mask.sum()),
        "row_to_vertex": "vertex_index.tsv",
        "design_source": {
            "driver": "mmmdata/scripts/glmsingle_tb.py (imported: discover_sessions, "
                      "load_all_events, build_condition_mapping, build_design_matrices)",
            "volume_fit": str(vol_dir),
            "checked_against_volume_fit": True,
        },
        "this_script": "mmmdata/scripts/functional_space/glmsingle_tb_fsaverage6.py",
        "mmmdata_git_sha": sha,
        "uncommitted_changes": {
            "glmsingle_tb.py": git_dirty(_SCRIPTS.parent, "scripts/glmsingle_tb.py"),
            "glmsingle_tb_fsaverage6.py": git_dirty(
                _SCRIPTS.parent, "scripts/functional_space/glmsingle_tb_fsaverage6.py"),
        },
        "tr": base.TR, "stimdur": base.STIMDUR_DEFAULT,
        "onset_to_volume": "round(onset / TR), as in glmsingle_tb.build_design_matrices",
        "confounds": "none (GLMdenoise handles denoising), as in the volume enc fit",
        "glmsingle_params": {**GLMSINGLE_PARAMS,
                             "sessionindicator": session_indices,
                             "xvalscheme": "GLMsingle default (one fold per run)"},
        "n_runs": len(run_list), "n_volumes_per_run": n_vols,
        "n_conditions": len(cond_map),
        "n_repeated_conditions": int((condition_key["n_presentations"] > 1).sum()),
        "n_trials": int(len(trial_info)),
        "run_labels": run_labels,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        (out_dir / "dry_run_manifest.json").write_text(json.dumps(sidecar, indent=2))
        print(f"DRY RUN — nothing fitted. {out_dir / 'dry_run_manifest.json'}")
        return

    try:
        from importlib.metadata import version
        sidecar["glmsingle_version"] = version("glmsingle")
    except Exception:
        sidecar["glmsingle_version"] = "unknown"

    dd = out_base / "dataset_description.json"
    if not dd.exists():
        dd.write_text(json.dumps({
            "Name": "GLMsingle single-trial betas, TB encoding, fsaverage6 cortex",
            "BIDSVersion": "1.8.0", "DatasetType": "derivative",
            "GeneratedBy": [
                {"Name": "GLMsingle", "Version": sidecar["glmsingle_version"],
                 "CodeURL": "https://github.com/cvnlab/GLMsingle"},
                {"Name": "glmsingle_tb_fsaverage6.py",
                 "Description": "mmmdata/scripts/functional_space/glmsingle_tb_fsaverage6.py; "
                                "design from mmmdata/scripts/glmsingle_tb.py (enc arm)"},
            ],
            "SourceDatasets": [{"URL": str(fmriprep_dir)}],
        }, indent=2))

    print("\nLoading surface BOLD...")
    data = []
    for i, (s, t, r) in enumerate(run_list):
        x = load_run(fmriprep_dir, args.subject, s, t, r, mask)
        if not np.isfinite(x).all():
            sys.exit(f"ERROR: non-finite values in {s}/{r}")
        flat = int((x.std(axis=1) == 0).sum())
        if flat:
            sys.exit(f"ERROR: {flat} constant cortex vertices in {s}/{r}")
        data.append(x)
    print(f"  {len(data)} runs x {data[0].shape[0]} vertices")

    from glmsingle.glmsingle import GLM_single
    params = {**GLMSINGLE_PARAMS,
              "sessionindicator": np.array(session_indices, dtype=int).reshape(1, -1)}
    glm = GLM_single(params)
    glm.fit(design=designs, data=data, stimdur=base.STIMDUR_DEFAULT, tr=base.TR,
            outputdir=str(out_dir / "glmsingle_outputs"),
            figuredir=str(out_dir / "glmsingle_figures"))

    condition_key.to_csv(out_dir / "condition_key.csv", index=False)
    trial_info.to_csv(out_dir / "trial_info.csv", index=False)
    vidx.to_csv(out_dir / "vertex_index.tsv", sep="\t", index=False)
    (out_dir / "run_metadata.json").write_text(json.dumps(sidecar, indent=2))
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
