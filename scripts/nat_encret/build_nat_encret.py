#!/usr/bin/env python3
"""Build the nat_encret derivative: NAT recall units, matcher maps, group event tables.

    build_nat_encret.py units [--out DIR]
        Cut every harmonised task-NATretrieval word table in beh/ into recall units and write one
        *_recallunits.tsv per table (row pointers into beh.tsv, no text), plus dataset_description.json.
    build_nat_encret.py map <work> [--matcher rmatch] [--revision SHA] [--out DIR]
        Package a collected matcher run (<work>/mapping_<matcher>.tsv from rmatch_run.py collect)
        as one *_desc-<matcher>_recallmap.tsv (+ provenance .json) per word table. Every unit of
        every word table the run touched must be covered, and each must match its unit table.
    build_nat_encret.py group [--matcher rmatch] [--out DIR]
        From the unit tables and maps: group/desc-<matcher>_eventrecall.tsv (participant x SEG-C
        event, first recalls) and group/desc-<matcher>_eventnorms.tsv (per event, pooled).
    build_nat_encret.py check [--out DIR]
        Rejoin every unit to its beh.tsv and re-cut it: the tables on disk must equal a fresh build.

Layout and conventions: encret_io.py. --out defaults to <config output_dir>/nat_encret.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import encret_io as io
import rmatch_run

UNIT_COLS = ["trial_num", "unit_idx", "stimulus_id", "recall_rank", "first_row", "last_row", "n_words",
             "onset", "offset"]
MAP_COLS = ["trial_num", "unit_idx", "stimulus_id", "seg_number", "mapped", "n_matched", "parse_ok", "hit_max"]

UNITS_JSON = {
    "Description": "Recall units of NAT spoken free recall: clause-sized spans of each recall trial's transcript, "
                   "as row ranges into this run's harmonised word table in the raw dataset "
                   "(sub-##/ses-##/beh/sub-##_ses-##_task-NATretrieval[_run-##]_beh.tsv). No transcript text is "
                   "stored here; rebuild a unit's text from those rows (behavioral.recall_units.rejoin in mmmdata).",
    "Segmentation": "Filled pauses (filler == 1) dropped; sentences end at . ! or ?; sentences over 40 words cut at "
                    ", ; : once a piece has 20 words, then into equal parts; units under 3 words join the previous "
                    "unit. See mmmdata src/python/behavioral/recall_units.py.",
    "trial_num": {"Description": "Recall trial, as trial_num in the beh.tsv and events.tsv."},
    "unit_idx": {"Description": "Unit order within the trial, from 0."},
    "stimulus_id": {"Description": "Film recalled (stimulus registry id, stimuli/stimulus_registry/movies.tsv)."},
    "recall_rank": {"Description": "Which recall of this film by this participant: 1 = first, ordered by session "
                                   "then trial across all of the participant's NATretrieval runs."},
    "first_row": {"Description": "First word of the unit: 0-based data-row index (header excluded) into the beh.tsv."},
    "last_row": {"Description": "Last word of the unit, inclusive, same indexing. Filled-pause rows inside "
                                "first_row..last_row are not part of the unit."},
    "n_words": {"Description": "Lexical words in the unit (filled pauses excluded)."},
    "onset": {"Description": "First word's onset on the run clock (as onset in the beh.tsv); n/a where that word's "
                             "onset is n/a.", "Units": "s"},
    "offset": {"Description": "Last word's onset + duration on the run clock; n/a where either is n/a.", "Units": "s"},
}

MAP_JSON = {
    "Description": "Recall unit -> SEG-C film event map from an automatic matcher: one row per (unit, matched "
                   "event); a unit matched to no event has one row with mapped = 0 and seg_number n/a.",
    "trial_num": {"Description": "Recall trial (joins the recallunits table)."},
    "unit_idx": {"Description": "Unit within the trial (joins the recallunits table)."},
    "stimulus_id": {"Description": "Film recalled; the events are this film's."},
    "seg_number": {"Description": "Matched SEG-C event: seg_number of level C in mmmdata "
                                  "scripts/parse_movie_annotations.py output for this film."},
    "mapped": {"Description": "1 if the unit matched at least one event.", "Levels": {"0": "no event", "1": "matched"}},
    "n_matched": {"Description": "Events this unit matched (0 when mapped = 0)."},
    "parse_ok": {"Description": "1 if the matcher's answer parsed; 0 rows count as no match (rMatch's fallback).",
                 "Levels": {"0": "unparsed", "1": "parsed"}},
    "hit_max": {"Description": "1 if generation stopped at the token limit.", "Levels": {"0": "no", "1": "yes"}},
}


def _norm_label(x) -> str:
    """'3' / 3 / 3.0 -> '03'; '' / NaN -> ''."""
    if x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip() in ("", "nan"):
        return ""
    return f"{int(float(x)):02d}"


def dataset_description(out: Path) -> None:
    io.write_json({
        "Name": "MMMData NAT encoding-retrieval coding",
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative",
        "Description": "Which parts of each NAT spoken film recall express which film events: recall units "
                       "(row pointers into the raw dataset's beh/ word tables, no transcript text), matcher maps "
                       "(desc-<matcher>) onto the films' SEG-C events, and group event tables.",
        "GeneratedBy": [{"Name": "mmmdata/scripts/nat_encret/build_nat_encret.py",
                         "Version": io.code_version()["commit"],
                         "CodeURL": "https://github.com/hulacon/mmmdata"}],
        "SourceDatasets": [{"URL": "../.."}, {"URL": "../stimuli_features"}],
    }, out / "dataset_description.json")


def write_units(U: pd.DataFrame, out: Path) -> list[Path]:
    U = io.round_times(U, ["onset", "offset"])
    written = []
    for (s, ses, run), g in U.groupby(["subject", "session", "run"], sort=True):
        p = io.units_path(out, dict(sub=s, ses=ses, run=run))
        io.write_tsv(g[UNIT_COLS], p)
        written.append(p)
    return written


def cmd_units(a):
    out = a.out or io.default_out()
    U = io.build_units(io.paths()["bids"])
    dataset_description(out)
    io.write_json(UNITS_JSON, out / f"task-{io.TASK}_recallunits.json")
    written = write_units(U, out)
    print(json.dumps(dict(out=str(out), files=len(written), n_units=len(U),
                          n_trials=int(U.groupby(["subject", "session", "trial_num"]).ngroups),
                          n_first_recall_units=int((U.recall_rank == 1).sum()))))


def load_mapping(work: Path, matcher: str) -> tuple[pd.DataFrame, dict]:
    M = pd.read_csv(work / f"mapping_{matcher}.tsv", sep="\t", dtype={"run": str}, keep_default_na=False,
                    na_values=[""])
    M["subject"] = M.subject.map(_norm_label)
    M["session"] = M.session.map(_norm_label)
    M["run"] = M.run.map(_norm_label)
    info_p = work / f"mapping_{matcher}.json"
    info = json.loads(info_p.read_text()) if info_p.exists() else {}
    return M, info


def map_provenance(work: Path, info: dict, revision: str | None) -> dict:
    shards = info.get("shards") or [json.loads(p.read_text()) for p in sorted((work / "shards").glob("gen_real_*of*.json"))
                                    if not p.stem.endswith(("smoke", "vsmoke"))]
    if not shards:
        raise SystemExit(f"no shard provenance under {work / 'shards'}")
    pick = lambda k: sorted({str(s.get(k)) for s in shards if s.get(k) is not None})  # noqa: E731
    models, engines, commits = pick("model"), pick("engine"), pick("rmatch_commit")
    revs = pick("revision") or ([revision] if revision else [])
    prompts = pick("prompt_sha256") or [rmatch_run.prompt_sha256()]
    for name, v in (("model", models), ("engine", engines), ("rmatch_commit", commits), ("revision", revs),
                    ("prompt_sha256", prompts)):
        if len(v) != 1:
            raise SystemExit(f"shards disagree on (or lack) {name}: {v}; pass --revision for runs that predate it")
    if revision and revs != [revision]:
        raise SystemExit(f"--revision {revision} contradicts the shards' {revs}")
    stamps = sorted(datetime.datetime.fromtimestamp(p.stat().st_mtime, datetime.UTC).date().isoformat()
                    for p in (work / "shards").glob("gen_real_*of*.parquet"))
    return {
        "Name": "rMatch",
        "CodeURL": "https://github.com/GabrielKP/rMatch",
        "Commit": commits[0],
        "Model": models[0],
        "ModelRevision": revs[0],
        "Engine": engines[0],
        "PromptSHA256": prompts[0],
        "Decoding": f"greedy, max_new_tokens {rmatch_run.MAX_NEW}",
        "RecallWindow": rmatch_run.WINDOW,
        "Story": "the film's SEG-C descriptions in time order, numbered 1..n",
        "GPU": pick("gpu"),
        "RunDate": stamps[-1] if stamps else None,
        "Units": {k: info[k] for k in ("n_units", "parse_fail", "out_of_range_ids", "hit_max",
                                       "share_units_mapped") if k in info},
    }


def cmd_map(a):
    out = a.out or io.default_out()
    M, info = load_mapping(a.work, a.matcher)
    if "source_film" in M and (M.source_film != M.film).any():
        raise SystemExit("mapping holds pseudo (cross-film) rows: only a real-condition run is packaged")
    U = io.read_units(out)
    subjects = sorted(M.subject.unique())
    # Coverage is checked per word table the run touched, so a run over new sessions only is
    # packaged without re-running the old ones; within a table, every unit must be mapped.
    files = M[["subject", "session", "run"]].drop_duplicates()
    U = U.merge(files, on=["subject", "session", "run"])
    key = ["subject", "session", "run", "trial", "unit_idx"]
    mu = M.drop_duplicates(key)
    j = U.rename(columns={"trial_num": "trial"}).merge(mu, on=key, how="outer", indicator=True,
                                                       suffixes=("", "_map"))
    lost, extra = j[j._merge == "left_only"], j[j._merge == "right_only"]
    if len(lost) or len(extra):
        raise SystemExit(f"map and unit tables disagree in {len(files)} word tables: {len(lost)} units unmapped, "
                         f"{len(extra)} mapped units not in the unit tables (rebuild units, or re-run the matcher)")
    both = j[j._merge == "both"]
    bad = (both.stimulus_id != both.film) | (both.n_words != both.n_words_map)
    if bad.any():
        raise SystemExit(f"{int(bad.sum())} units differ between map and unit tables (film or n_words)")
    prov = map_provenance(a.work, info, a.revision)
    T = M.rename(columns={"trial": "trial_num", "film": "stimulus_id", "event_seg_number": "seg_number"})
    for c in ("mapped", "parse_ok", "hit_max"):
        T[c] = T[c].astype(str).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0}).astype(int)
    T["seg_number"] = T.seg_number.astype("Int64")
    io.write_json(MAP_JSON, out / f"task-{io.TASK}_desc-{a.matcher}_recallmap.json")
    n = 0
    for (s, ses, run), g in T.groupby(["subject", "session", "run"], sort=True):
        ent = dict(sub=s, ses=ses, run=run)
        p = io.map_path(out, ent, a.matcher)
        io.write_tsv(g.sort_values(["trial_num", "unit_idx", "seg_number"])[MAP_COLS], p)
        io.write_json({"Matcher": prov}, p.with_suffix(".json"))
        n += 1
    dataset_description(out)
    print(json.dumps(dict(out=str(out), matcher=a.matcher, files=n, n_units=len(mu), subjects=subjects,
                          revision=prov["ModelRevision"])))


def read_maps(out: Path, matcher: str) -> pd.DataFrame:
    files = sorted(out.glob(f"sub-*/ses-*/beh/sub-*_task-{io.TASK}*_desc-{matcher}_recallmap.tsv"))
    if not files:
        raise FileNotFoundError(f"no desc-{matcher} recallmap tables under {out}")
    parts = []
    for f in files:
        ent = io.entities(f.name)
        parts.append(io.read_tsv(f).assign(subject=ent["sub"], session=ent["ses"], run=ent.get("run", "")))
    return pd.concat(parts, ignore_index=True)


def event_grid(ev: pd.DataFrame, segb: pd.DataFrame) -> pd.DataFrame:
    """SEG-C events with film position and their SEG-B scene (the scene holding the event's midpoint,
    else the scene whose midpoint is nearest)."""
    c = ev[["film", "seg_number", "onset", "offset"]].copy()
    c["mid"] = (c.onset + c.offset) / 2
    c = c.join(c.groupby("film").offset.max().rename("film_end"), on="film")
    c["position"] = c.mid / c.film_end
    out = []
    for f, g in c.groupby("film"):
        bb = segb[segb.film == f].sort_values("onset")
        for mid in g.mid:
            inside = bb[(bb.onset <= mid) & (mid < bb.offset)]
            if len(inside):
                out.append(int(inside.seg_number.iloc[0]))
            else:
                out.append(int(bb.seg_number.iloc[np.argmin(np.abs((bb.onset + bb.offset) / 2 - mid))]))
    c["segb"] = out
    return c


def event_recall(U: pd.DataFrame, Mp: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Participant x SEG-C event for each first recall: units and words mapped to it, recalled 0/1."""
    m = Mp.merge(U[["subject", "session", "run", "trial_num", "unit_idx", "recall_rank", "n_words"]],
                 on=["subject", "session", "run", "trial_num", "unit_idx"], how="left", validate="m:1")
    r1 = m[m.recall_rank == 1]
    trials = r1[["subject", "stimulus_id", "session", "trial_num"]].drop_duplicates()
    full = trials.merge(grid.rename(columns={"film": "stimulus_id"}), on="stimulus_id")
    hit = (r1[r1.mapped == 1].groupby(["subject", "stimulus_id", "seg_number"])
           .agg(n_units=("unit_idx", "size"), n_words=("n_words", "sum")).reset_index())
    hit["seg_number"] = hit.seg_number.astype(int)
    full = full.merge(hit, on=["subject", "stimulus_id", "seg_number"], how="left")
    full[["n_units", "n_words"]] = full[["n_units", "n_words"]].fillna(0).astype(int)
    full["recalled"] = (full.n_units > 0).astype(int)
    # A map has one row per (unit, matched event): totals count each unit once. A unit matched to
    # several events adds its words to each of them (detail per event, not a partition).
    uniq = r1.drop_duplicates(["subject", "stimulus_id", "session", "trial_num", "unit_idx"])
    tot = uniq.groupby(["subject", "stimulus_id"]).agg(trial_words=("n_words", "sum"),
                                                        trial_units=("unit_idx", "size"))
    full = full.join(tot, on=["subject", "stimulus_id"])
    return full.sort_values(["stimulus_id", "subject", "onset"]).reset_index(drop=True)


def event_norms(P: pd.DataFrame) -> pd.DataFrame:
    g = P.groupby(["stimulus_id", "seg_number"], sort=False)
    N = g.agg(onset=("onset", "first"), offset=("offset", "first"), position=("position", "first"),
              segb=("segb", "first"), n_subjects=("subject", "nunique"), n_recalled=("recalled", "sum"),
              mean_words=("n_words", "mean"), mean_units=("n_units", "mean")).reset_index()
    N["p_recall"] = N.n_recalled / N.n_subjects
    cols = ["stimulus_id", "seg_number", "onset", "offset", "position", "segb", "n_subjects", "n_recalled",
            "p_recall", "mean_words", "mean_units"]
    return N[cols]


GROUP_JSON = {
    "eventrecall": {
        "Description": "Participant x SEG-C event, first recall of each film (recall_rank 1), from the "
                       "desc-{m} recallmap tables. One row per event of each film the participant recalled.",
        "seg_number": {"Description": "SEG-C event (parse_movie_annotations.py, level C)."},
        "mid": {"Description": "Event midpoint in film time.", "Units": "s"},
        "film_end": {"Description": "Last SEG-C offset of the film.", "Units": "s"},
        "position": {"Description": "mid / film_end."},
        "segb": {"Description": "SEG-B scene holding the event's midpoint (else the nearest scene)."},
        "n_units": {"Description": "Units of this recall mapped to the event."},
        "n_words": {"Description": "Words of those units; a unit mapped to k events counts toward each."},
        "recalled": {"Description": "1 if n_units > 0."},
        "trial_words": {"Description": "Lexical words in the whole first recall."},
        "trial_units": {"Description": "Units in the whole first recall."},
    },
    "eventnorms": {
        "Description": "Per SEG-C event, first recalls pooled over participants (from desc-{m}_eventrecall.tsv).",
        "n_subjects": {"Description": "Participants with a first recall of the film."},
        "n_recalled": {"Description": "Of those, participants whose first recall mapped a unit to the event."},
        "p_recall": {"Description": "n_recalled / n_subjects."},
        "mean_words": {"Description": "Mean n_words over n_subjects (0 for a participant who did not recall it)."},
        "mean_units": {"Description": "Mean n_units over n_subjects."},
    },
}


def cmd_group(a):
    out = a.out or io.default_out()
    U, Mp = io.read_units(out), read_maps(out, a.matcher)
    ann = io.annotations()
    P = event_recall(U, Mp, event_grid(io.segc_events(ann), io.segb_spans(ann)))
    N = event_norms(P)
    g = out / "group"
    io.write_tsv(P, g / f"desc-{a.matcher}_eventrecall.tsv")
    io.write_tsv(N, g / f"desc-{a.matcher}_eventnorms.tsv")
    for k, js in GROUP_JSON.items():
        js = json.loads(json.dumps(js).replace("{m}", a.matcher))
        io.write_json(js, g / f"desc-{a.matcher}_{k}.json")
    print(json.dumps(dict(out=str(g), matcher=a.matcher, eventrecall_rows=len(P), events=len(N),
                          subjects=sorted(P.subject.unique()))))


def cmd_check(a):
    out = a.out or io.default_out()
    bids = io.paths()["bids"]
    disk = io.read_units(out)
    fresh = io.round_times(io.build_units(bids), ["onset", "offset"])
    key = io.UNIT_KEY
    j = fresh.merge(disk, on=key, how="outer", indicator=True, suffixes=("", "_disk"))
    problems = []
    if (j._merge != "both").any():
        problems.append(f"{int((j._merge != 'both').sum())} units on one side only")
    b = j[j._merge == "both"]
    for c in UNIT_COLS:
        if c in key:
            continue
        x, y = b[c], b[f"{c}_disk"]
        same = (x == y) | (x.isna() & y.isna())
        if c in ("onset", "offset"):
            same = np.isclose(x.astype(float), y.astype(float), atol=1e-6, equal_nan=True)
        if not np.all(same):
            problems.append(f"{c}: {int((~np.asarray(same)).sum())} rows differ")
    print(json.dumps(dict(out=str(out), n_units=len(disk), problems=problems)))
    return 1 if problems else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="verb", required=True)
    for name in ("units", "map", "group", "check"):
        p = sp.add_parser(name)
        p.add_argument("--out", type=Path, default=None, help="nat_encret root (default: config output_dir)")
        if name == "map":
            p.add_argument("work", type=Path, help="rmatch_run.py work dir holding mapping_<matcher>.tsv")
            p.add_argument("--revision", default=None,
                           help="model revision, for runs whose shards predate recording it")
        if name in ("map", "group"):
            p.add_argument("--matcher", default="rmatch")
    a = ap.parse_args(argv)
    return dict(units=cmd_units, map=cmd_map, group=cmd_group, check=cmd_check)[a.verb](a) or 0


if __name__ == "__main__":
    sys.exit(main())
