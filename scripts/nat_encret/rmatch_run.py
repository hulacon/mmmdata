#!/usr/bin/env python3
"""rMatch's prompt, run with a local LLM through vLLM: NAT recall units -> SEG-C film events.

rMatch (github.com/GabrielKP/rMatch) asks an LLM which numbered story pieces a
target recall clause expresses, given the whole story and +-5 clauses of recall
around it. Here the story pieces are a film's SEG-C event descriptions and the
clauses are the nat_encret recall units.

    rmatch_run.py prompts  <work> [--condition real|pseudo] [--first-only] [--subjects S ...] [--films F ...]
    rmatch_run.py vgenerate <work> --condition C [--task I --n-tasks N] [--limit K]   (GPU; rmatch_vllm.sbatch)
    rmatch_run.py collect  <work> --condition C

``<work>`` is a scratch directory for one run (prompts, shards, the collected
mapping). ``build_nat_encret.py map <work>`` then packages the collected
``mapping_rmatch.tsv`` into the derivative. Unit text is rebuilt from ``beh/``
at prompt time, so the prompts file holds participant speech: keep ``<work>``
out of any shared or public tree.

The prompt template, the parser and the two formatting helpers are copied
verbatim from rMatch @ f86b543 (src/rmatch/matchers/matcher_base.py and
src/rmatch/utils.py), not imported, because importing ``rmatch`` pulls in its
API-client dependencies. Decoding follows rMatch's MatcherCuda: greedy,
max_new_tokens 1024, recall window +-5.

Conditions:
  real    each recall trial against its own film's events (``--first-only``: first recalls only).
  pseudo  matcher-only control: participant s's first recall of a seeded, randomly drawn OTHER film,
          against film f's events, for every (s, f). It has nothing to do with f, so whatever it maps,
          and any between-participant agreement in that, comes from the matcher.

Output of collect: ``mapping_rmatch[-pseudo].tsv``, one row per (unit, matched event); a unit
matching nothing gets one row with mapped False.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

import encret_io as io

MODEL = "google/gemma-4-31B-it"
RMATCH_COMMIT = "f86b543"
WINDOW, MAX_NEW = 5, 1024
MAX_MODEL_LEN = 8192
SEED_PSEUDO = 18

# ---- verbatim from rMatch @ f86b543 (matcher_base.py) ----
DEFAULT_PROMPT = """This is the original story:
{story_full}

It can be broken down into the following independent pieces of information:
{story_segments}

Below is a window of consecutive clauses from a participant's recall.
The TARGET clause is marked with >>> <<<.
The other clauses are provided _only as context_.

Recall window:
{recall_window}

Which of the numbered information pieces are
expressed by the >>> TARGET <<< clause?

Use the surrounding clauses only to resolve references (e.g., pronouns),
but DO NOT attribute information expressed only in neighboring clauses.
If a numbered information piece is not explicitly expressed in the TARGET
clause itself, do NOT include it.

If none apply, return "<NONE>".

First reason about the TARGET clause and which story elements it maps to.

Then return a set of numbers in angle brackets, for example:
<3, 7, 12>"""


def _parse_output(raw: str) -> set[int] | None:
    raw = raw.strip()
    if "<NONE>" in raw:
        return set()
    match = re.search(r"<([0-9,\s]+)>", raw)
    if not match:
        return None
    return {int(x.strip()) for x in match.group(1).split(",") if x.strip().isdigit()}


# ---- verbatim from rMatch @ f86b543 (utils.py) ----
def format_story_segments(story_segments: list[str]) -> str:
    return "\n".join(f"{i + 1}. {seg.strip()}" for i, seg in enumerate(story_segments))


def build_recall_window(recall_segments, target_idx, window_size, start_marker="<target>", end_marker="</target>"):
    if window_size == 0:
        return f"{start_marker}{recall_segments[target_idx]}{end_marker}"
    start = max(0, target_idx - window_size)
    end = min(len(recall_segments), target_idx + window_size + 1)
    lines = []
    for i in range(start, end):
        clause = recall_segments[i].strip()
        lines.append(f"{start_marker}{clause}{end_marker}" if i == target_idx else f"{clause}")
    return "\n".join(lines)


def build_prompt(story: list[str], recall: list[str], idx: int) -> str:  # MatcherBase.build_model_prompt
    p = DEFAULT_PROMPT
    for slot, val in (("{story_full}", " ".join(s.strip() for s in story)),
                      ("{story_segments}", format_story_segments(story)),
                      ("{recall_window}", build_recall_window(recall, idx, WINDOW, ">>> ", " <<<"))):
        p = p.replace(slot, val)
    return p
# ---- end verbatim ----


def prompt_sha256() -> str:
    return hashlib.sha256(DEFAULT_PROMPT.encode()).hexdigest()


def make_prompts(units: pd.DataFrame, events: pd.DataFrame, condition: str = "real") -> pd.DataFrame:
    """One prompt per (trial, unit, target film). ``units`` needs a ``text`` column."""
    story = {f: g.text.tolist() for f, g in events.groupby("film")}
    U = units.rename(columns={"trial_num": "trial", "stimulus_id": "film"})
    trials = U.groupby(["subject", "session", "trial", "film", "recall_rank"], sort=True)
    if condition == "real":
        pairs = [(k, k[3]) for k in trials.groups]  # (trial key, target film)
    else:
        rng = np.random.default_rng(SEED_PSEUDO)
        first = {(k[0], k[3]): k for k in trials.groups if k[4] == 1}
        pairs = []
        for s in sorted({k[0] for k in first}):
            films = sorted(f for (ss, f) in first if ss == s)
            for f in films:
                src = rng.choice([g for g in films if g != f])
                pairs.append((first[(s, src)], f))
    missing = sorted({t for _, t in pairs} - set(story))
    if missing:
        raise ValueError(f"recalled films without SEG-C events: {missing}")
    rows = []
    for key, target in pairs:
        g = trials.get_group(key).sort_values("unit_idx")
        rec = g.text.tolist()
        for j, (_, u) in enumerate(g.iterrows()):
            rows.append(dict(subject=key[0], session=key[1], trial=key[2], source_film=key[3], recall_rank=key[4],
                             film=target, unit_idx=int(u.unit_idx), n_words=int(u.n_words),
                             unit_onset=float(u.onset), unit_offset=float(u.offset), run=u.run,
                             n_events=len(story[target]), prompt=build_prompt(story[target], rec, j)))
    return pd.DataFrame(rows)


def cmd_prompts(a):
    p = io.paths()
    U = io.read_units(a.deriv or io.default_out(), subjects=a.subjects)
    if a.first_only:
        U = U[U.recall_rank == 1]
    if a.films:
        missing = set(a.films) - set(U.stimulus_id)
        if missing:
            raise SystemExit(f"--films not among the units: {sorted(missing)}")
        U = U[U.stimulus_id.isin(a.films)]
    P = make_prompts(io.with_text(p["bids"], U), io.segc_events(), a.condition)
    a.work.mkdir(parents=True, exist_ok=True)
    P.to_parquet(a.work / f"prompts_{a.condition}.parquet")
    print(json.dumps(dict(condition=a.condition, n_prompts=len(P),
                          n_trials=int(P.groupby(["subject", "session", "trial", "film"]).ngroups),
                          prompt_chars_median=int(P.prompt.str.len().median()))))


def cmd_vgenerate(a):
    """vLLM route: rMatch's MatcherCuda settings (greedy, max_tokens 1024, gpu_memory_utilization .90,
    trust_remote_code). One difference: max_model_len 8192, not 90000, because the longest prompt + 1024 is
    < 7k tokens and 90k would not fit beside the 62 GB weights. Prompts are sharded by FILM, so each shard
    holds whole films and vLLM's prefix cache can reuse a film's story text across its recall units.
    Each shard is written only at the end, so a preempt requeue reruns it whole."""
    from vllm import LLM, SamplingParams
    P = pd.read_parquet(a.work / f"prompts_{a.condition}.parquet")
    films = sorted(P.film.unique())
    mine_f = set(films[a.task::a.n_tasks])
    mine = np.flatnonzero(P.film.isin(mine_f).to_numpy())
    if a.limit:
        mine = mine[np.linspace(0, len(mine) - 1, a.limit).astype(int)]
    t0 = time.time()
    llm = LLM(model=MODEL, tensor_parallel_size=1, gpu_memory_utilization=0.90, max_model_len=MAX_MODEL_LEN,
              trust_remote_code=True, enable_prefix_caching=True, seed=0)
    tok = llm.get_tokenizer()
    load_s = round(time.time() - t0, 1)
    texts = [tok.apply_chat_template([{"role": "user", "content": P.prompt.iloc[i]}], tokenize=False,
                                     add_generation_prompt=True) for i in mine]
    print(json.dumps(dict(load_s=load_s, n=len(mine), films=len(mine_f))), flush=True)
    t0 = time.time()
    outs = llm.generate(texts, SamplingParams(max_tokens=MAX_NEW, temperature=0.0))
    gen_s = round(time.time() - t0, 1)
    R = P.iloc[mine].drop(columns="prompt").assign(
        raw=[o.outputs[0].text.strip() for o in outs],
        n_prompt_tokens=[len(o.prompt_token_ids) for o in outs],
        n_out_tokens=[len(o.outputs[0].token_ids) for o in outs],
        n_cached_tokens=[getattr(o, "num_cached_tokens", None) or 0 for o in outs])
    R["hit_max"] = R.n_out_tokens >= MAX_NEW
    tag = f"{a.condition}_{a.task:02d}of{a.n_tasks:02d}" + ("_vsmoke" if a.limit else "")
    (a.work / "shards").mkdir(exist_ok=True)
    R.to_parquet(a.work / "shards" / f"gen_{tag}.parquet")
    import torch
    import vllm
    info = dict(model=MODEL, revision=_cached_revision(MODEL), engine=f"vllm {vllm.__version__}",
                rmatch_commit=RMATCH_COMMIT, prompt_sha256=prompt_sha256(), condition=a.condition,
                task=a.task, n_tasks=a.n_tasks, films=sorted(mine_f), n=len(R), load_s=load_s, gen_s=gen_s,
                prompt_tokens=int(R.n_prompt_tokens.sum()), cached_tokens=int(R.n_cached_tokens.sum()),
                out_tokens=int(R.n_out_tokens.sum()), hit_max=int(R.hit_max.sum()),
                gpu=torch.cuda.get_device_name(0), mmmdata=io.code_version())
    (a.work / "shards" / f"gen_{tag}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "films"}), flush=True)


def _cached_revision(model: str) -> str | None:
    """The snapshot the offline HF cache resolves ``model`` to (refs/main), or None."""
    import os
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    ref = hub / f"models--{model.replace('/', '--')}" / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else None


def collect(work: Path, condition: str, events: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    fs = sorted(glob.glob(str(work / "shards" / f"gen_{condition}_*of*.parquet")))
    fs = [f for f in fs if not f.endswith(("_smoke.parquet", "_vsmoke.parquet"))]
    if not fs:
        raise SystemExit(f"no shards for {condition} in {work / 'shards'}")
    R = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    P = pd.read_parquet(work / f"prompts_{condition}.parquet", columns=["subject", "film", "unit_idx", "session",
                                                                         "trial", "source_film"])
    key = ["subject", "session", "trial", "source_film", "film", "unit_idx"]
    if R.duplicated(key).any():
        raise SystemExit("duplicate units across shards")
    missing = len(P) - len(R)
    if missing:
        raise SystemExit(f"{missing} prompts have no output: a shard is missing or incomplete")
    seg = {f: g.reset_index(drop=True) for f, g in events.groupby("film")}
    rows, n_fail, n_oob = [], 0, 0
    for _, r in R.iterrows():
        parsed = _parse_output(r.raw)
        base = dict(subject=r.subject, session=r.session, run=r.run, trial=r.trial, unit_idx=r.unit_idx,
                    film=r.film, source_film=r.source_film, recall_rank=r.recall_rank, unit_onset=r.unit_onset,
                    unit_offset=r.unit_offset, n_words=r.n_words, parse_ok=parsed is not None,
                    hit_max=bool(r.hit_max))
        if parsed is None:
            n_fail += 1
            parsed = set()  # rMatch's fallback after its retries: no match
        ok = sorted(i for i in parsed if 1 <= i <= r.n_events)
        n_oob += len(parsed) - len(ok)
        if not ok:
            rows.append(dict(base, event_seg_number=np.nan, event_onset=np.nan, event_offset=np.nan, mapped=False,
                             n_matched=0))
        for i in ok:
            e = seg[r.film].iloc[i - 1]
            rows.append(dict(base, event_seg_number=int(e.seg_number), event_onset=float(e.onset),
                             event_offset=float(e.offset), mapped=True, n_matched=len(ok)))
    M = pd.DataFrame(rows).sort_values(["subject", "session", "trial", "unit_idx", "event_onset"])
    u = M.drop_duplicates(["subject", "session", "trial", "film", "unit_idx"])
    shards = [json.loads(Path(f).with_suffix(".json").read_text()) for f in fs]
    info = dict(condition=condition, n_units=len(u), parse_fail=n_fail, out_of_range_ids=n_oob,
                hit_max=int(u.hit_max.sum()), share_units_mapped=round(float(u.mapped.mean()), 4),
                events_per_mapped_unit=round(float(M[M.mapped].groupby(
                    ["subject", "session", "trial", "unit_idx"]).size().mean()), 3),
                shards=[{k: v for k, v in s.items() if k != "films"} for s in shards])
    return M, info


def cmd_collect(a):
    M, info = collect(a.work, a.condition, io.segc_events())
    name = "rmatch" if a.condition == "real" else "rmatch-pseudo"
    M.to_csv(a.work / f"mapping_{name}.tsv", sep="\t", index=False)
    (a.work / f"mapping_{name}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "shards"}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="verb", required=True)
    p = sp.add_parser("prompts")
    p.add_argument("work", type=Path)
    p.add_argument("--deriv", type=Path, default=None, help="nat_encret root (default: config output_dir)")
    p.add_argument("--condition", choices=["real", "pseudo"], default="real")
    p.add_argument("--first-only", action="store_true", help="first recall of each film only")
    p.add_argument("--subjects", nargs="+", default=None, help="bare labels, e.g. 06 07")
    p.add_argument("--films", nargs="+", default=None)
    v = sp.add_parser("vgenerate")
    v.add_argument("work", type=Path)
    v.add_argument("--condition", choices=["real", "pseudo"], required=True)
    v.add_argument("--task", type=int, default=0)
    v.add_argument("--n-tasks", type=int, default=1)
    v.add_argument("--limit", type=int, default=0)
    c = sp.add_parser("collect")
    c.add_argument("work", type=Path)
    c.add_argument("--condition", choices=["real", "pseudo"], required=True)
    a = ap.parse_args(argv)
    dict(prompts=cmd_prompts, vgenerate=cmd_vgenerate, collect=cmd_collect)[a.verb](a)


if __name__ == "__main__":
    main()
