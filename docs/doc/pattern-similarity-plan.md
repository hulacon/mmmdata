# Pattern-Similarity Analysis: Preprocessing × Timeseries × Paradigm Comparison

Status: **Phases 0–5 complete and verified; Phase 6 code written, full-grid job running** (2026-08-11). Phase 2 NAT fits: array **46044299** COMPLETED (6/6, 13.3–14.4 h, masked-2D ~258k voxels × 4309–4310 chunk presentations); verified by new `scripts/pattern_similarity/verify_natfits.py` — 45/45 checks pass (115 repeated conditions ×10 from the 2 repeated movies, chunk_info consistent + identical across pipelines, betas trailing dims match, EVC split-half reliability of repeated-movie chunk betas mean r = 0.16–0.26, all positive; report at `qc/phase2_natfit_verification.tsv`). Phase 3 rawTR: array **46044912** COMPLETED; `extract_rawtr.py --verify` passes (62/62 caches per sub×pipeline, TB 61 trials, NAT TR sums match events, residual⊥confound spot-checks ~1e-13). Phase 4 TB caches: array **46044913** COMPLETED (2562 trials; ROI voxel counts EVC 1401, EAC 604, Hipp 1391, AG 2409, Precuneus 5623, mPFC 976). Phase 5 NAT caches: array **46056755** COMPLETED (~33 s/task) and verified (cols == chunk_info rows; repeated chunks exactly 115×10 in the index; voxel counts match Phase 3/4; sub-03 mPFC has 5 out-of-brain-mask NaN voxels — intended convention). **Phase 6**: `scripts/pattern_similarity/similarity.py` + `.sbatch` written; smoke-tested on TB/original/glmsingle (within_same > within_diff in all 6 ROIs; across-subject cells near zero with same > diff in 5/6 ROIs). Note discovered during implementation: TB triplet occurrences are contiguous 3-trial blocks but *unevenly distributed* across a session's 3 runs (0–3 occurrences per run, e.g. sub-03 ses-04 run-02 has two) — `tb_structure()` handles multi-block runs. Surrogate draws are seeded per (paradigm, cell, unit, item) excluding pipeline/timeseries, so pipeline contrasts are exactly paired. Full-grid job **46056771** (single job, 4 CPU/32 G/2 h) → `results/similarity_summary.tsv` + `similarity_cells.tsv`. **Next**: check 46056771 output + closed-form pair-count checks (819/1764/126 TB; 45/100/10 NAT), then Phase 7 `report.py`.

## Context

Goal: a comprehensive pattern-similarity analysis comparing preprocessing setups and encoding paradigms in MMMData. The full factorial design:

- **Pipelines**: fmriprep vs fmriprep_nordic
- **Timeseries**: rawTR (Model 8 confounds regressed) vs GLMsingle betas
- **ROIs (6, Harvard-Oxford, bilateral)**: early visual cortex (EVC), early auditory cortex (EAC), hippocampus, angular gyrus (AG), precuneus, medial PFC
- **Paradigms**: TBencoding vs NATencoding
- **Similarity cells (4)**: within-subject across-session same item; within-subject across-session different item; across-subject same item; across-subject different item

Metric: ROI pattern similarity = Pearson r across voxels, averaged across relevant trials/TRs/chunks.

Same items: NAT = the 2 movies repeated every session ("The Bench" 238 s, "From Dad To Son" 288 s; `condition==3` in events); TB = the repeated triplet (`sharedId==1 & enCon==3`, pairId 1–3, presented 3×/session × 14 sessions = 42 presentations/item). Different items: NAT = random other movie from the same run subsampled to same TR count; TB = random other triplet of items from the same run.

**Decisions locked in by Ben:**
1. NAT gets a **new GLMsingle fit** with betas every 4.5 s (movies chunked into 4.5-s pseudo-trials) so the 2×2 fully crosses with paradigm.
2. TB unit = per-trial patterns, same-item matched by `mmmId`.
3. Confounds = Model 8 (6 motion + CSF + WM + global signal + cosine; winner of the 21-model ISC benchmark) applied identically to rawTR streams of both paradigms; GLMsingle streams get no external confounds (GLMdenoise) — also identical across paradigms.
4. ROIs = Harvard-Oxford for all six.

**Deliverables**: analysis code in this repo (`scripts/pattern_similarity/` + `scripts/glmsingle_natencoding.py`); results under `derivatives/pattern_similarity/`.

## Verified facts (exploration + direct checks, 2026-08-10)

- Subjects sub-03/04/05, complete, identical coverage. TR = 1.5 s everywhere. Only volumetric space on disk: `MNI152NLin2009cAsym_res-2` (no T1w-space BOLD). All 609 preprocessing-QC run decisions are "keep" — no exclusions.
- **TBencoding**: ses-04..17, 14 sessions × 3 runs, 210 vols/run, 61 image trials/run, 3 s stim, 4.5 s SOA. GLMsingle TYPED betas exist for both pipelines: `derivatives/glmsingle{,_nordic}/sub-*/glmsingle_outputs/TYPED_FITHRF_GLMDENOISE_RR.npy` (11.8 GB, `betasmd` (97,115,97,2562)) + `trial_info.csv` + `condition_key.csv`, index-aligned across pipelines. The prior TB fit peaked at **200 GB RAM / 18.3 h on 8 CPUs** (sacct 43272306; 128 GB OOM'd) — sizes the NAT fits.
- **NATencoding**: ses-19..28, 10 sessions × 2 runs, 550–841 vols, 4 movies/run (ses-28 has 3). Repeated-movie durations stable ±0.02 s.
- **HO labels verified**: cort-maxprob-thr25-2mm — EVC=24 (Intracalcarine), EAC=45 (Heschl's), AG=21, Precuneus=31, mPFC=25 (Frontal Medial); subcortical 9+19 = hippocampus L+R. mPFC/precuneus are midline → bilateral ROIs (lateralized split available as an optional flag).
- **Shared-item structure (RESOLVED 2026-08-10 — intentional design, not a bug)**: every subject sees the same 6 shared items (mmmId 995 teacher, 996 remark, 997 attend, 998 theater, 999 create, 1000 angel), each 42× (3/session × 14 sessions). A per-subject seed in the design notebook (`mmmsourcedata/shared/experiment_code/cued_recall/stimuli.ipynb`, cell "Shuffle the 6 pairs") shuffles the pool: first 3 → that subject's contiguous **triplet** (pairId 1–3, `enCon==3`), last 3 → **super-repeat singles** (pairId 75–77, `enCon==2`, presented individually). Triplets as run: sub-03 = theater/angel/create (998/1000/999); sub-04 = create/angel/theater; sub-05 = angel/attend/create (997 in triplet, 998 in singles). Verified end-to-end: master design `cued_recall_encoding_seq.csv` (10 pre-generated subjects) → acquisition CSVs (`mmmsourcedata/sub-XX/ses-YY/behavioral/`) → BIDS events all agree. **Consequence (Ben's decision): across-subject TB same-item cell uses all 6 shared items matched by mmmId, with a `context_match` column (triplet–triplet, single–single, mixed); within-subject cells stay triplet-only per the original spec.**
- **Env**: `code/mmmdata/.venv` (Python 3.11.13, glmsingle installed). SLURM on Talapas; never use `--mail-type`.

### Reusable code (this repo)
- `scripts/nordic_benchmark/shared.py` — constants (BIDS_ROOT, DERIV_ROOT, SUBJECTS, TB_SESSIONS, NAT_SESSIONS, TR, MNI_SPACE, PIPELINES, GLMSINGLE_DIRS), path helpers (`bold_path/mask_path/events_path/confounds_path`), `read_tb_trials()`, `load_trial_metadata()`, HO `load_roi_masks()` + `resample_masks_to_bold()`.
- `scripts/nordic_benchmark/tb_eval.py` — `load_glmsingle_betas()`, finite-voxel + `np.corrcoef` pair logic in `session_pair_stats()` (adapt for across-session pairs).
- `scripts/nordic_benchmark/nat_eval.py` — `regress_confounds()` (OLS lstsq, works on (T,V) targets as-is), `confound_matrix_for_run()` (Model 8 via vendored `code/isc-confounds/extract_confounds.py` + `scripts/isc_confounds/model_meta.json` key "8"), natsort shim.
- `scripts/glmsingle_tbencoding.py` — template for the NAT GLMsingle producer.
- `scripts/glmsingle_qc.py` — `compute_voxel_reliability()` (Prince split-half) for the Phase 2 sanity check.
- `scripts/nordic_benchmark/report.py` + `report_html.py` — aggregation, figures, Wilcoxon, self-contained HTML scaffolding.

## Resolved design defaults

| # | Decision | Choice |
|---|----------|--------|
| D1 | ROI laterality | Bilateral (6 ROIs); optional `--split-hemi` |
| D2 | HRF lag (rawTR) | +4.5 s shift in both paradigms. TB trial pattern = mean of vols `onset_tr+3, onset_tr+4`; NAT = per-TR patterns shifted +3 TRs |
| D3 | Similarity unit | NAT rawTR: spatial r per matched TR pair, averaged over movie TRs. NAT GLMsingle: per-chunk beta patterns. TB: per-trial patterns. Matches "averaged across trials/TRs" |
| D4 | Different-item baseline | N=10 seeded random draws per same-item pair (`np.random.default_rng(20260810)`), matched TR/trial count, from the same run as the second pair member; n_draws recorded |
| D5 | Across-subject session pairing | All session pairs (10×10 NAT; all cross-subject presentation pairs TB); same-session-number subset also reported (`session_scope=matched`) as robustness |
| D6 | Voxel selection | All in-ROI ∩ brain-mask voxels, finite-value filter per pair (tb_eval convention); reliability-thresholding noted as extension |
| D7 | NAT chunking | `n_chunks = floor(duration/4.5)` (tail dropped); 4.5 s = exactly 3 TRs so chunk onset vol = `round(movie_onset/1.5) + 3*chunk_idx` (≤0.75 s onset jitter, identical across pipelines). Repeated movies share condition columns across sessions (52+63 = 115 conditions × 10 reps — these give GLMsingle its needed repeats; verified: "The Bench" 158 TRs/52 chunks, "From Dad To Son" 191 TRs/63 chunks since durations are fractionally under 288 s); non-repeated movies' chunks are one-shot conditions (~3,280 total). `stimdur=4.5` |
| D8 | Output naming | `derivatives/glmsingle_nat/{sub}/` + `glmsingle_nat_nordic/{sub}/` mirroring existing layout (`chunk_info.csv` analog of trial_info.csv); analysis outputs under `derivatives/pattern_similarity/{cache,results,qc,figures,report}/` |

## Implementation phases

### Phase 0 — Input verification (Task 0)
**Script**: `scripts/pattern_similarity/verify_inputs.py` (read-only; writes `derivatives/pattern_similarity/qc/task0_verification.tsv`). Login node, <5 min. (The triplet question is already resolved — see above — so this phase is pure mechanical verification, no gate.)
1. Enumerate all `sharedId==1` image rows for all subjects × 14 sessions × 3 runs; assert each subject's pairId→mmmId mapping (triplet 1–3 + singles 75–77) is constant across all sessions and each of the 6 items totals 42 presentations; write the per-subject shared-item registry (mmmId, word, role, triplet position) that Phase 6 consumes.
2. Verify repeated movies: `condition==3`, 1 presentation/session, durations ±0.05 s; record TR/chunk counts (verified: 158 TRs/52 chunks "The Bench"; 191/63 "From Dad To Son").
3. Verify every NAT run has ≥2 movies (needed for same-run different-item sampling; ses-28 is the irregular session).
4. Print bilateral HO ROI voxel counts on the res-2 grid.

### Phase 1 — Package scaffolding
**Create** `scripts/pattern_similarity/shared.py`: re-export from `nordic_benchmark/shared.py`; add `PATTERN_ROIS` (6 bilateral HO ROIs, labels above) + `load_bilateral_roi_masks()`; `GLMSINGLE_NAT_DIRS`; `PS_ROOT = DERIV_ROOT/"pattern_similarity"`; `HRF_SHIFT_TRS=3`, `SEED=20260810`, `N_DIFF_DRAWS=10`; repeated-item registries discovered from events (not hardcoded); result schema `RESULT_COLS = [paradigm, pipeline, timeseries, roi, cell, unit, item, r, n_pairs, n_voxels, n_draws, confound_model, session_scope, context_match]` with `cell ∈ {within_same, within_diff, across_same, across_diff}`, `unit` = subject or subject-pair.
**Verify**: imports clean from the venv; ROI voxel counts sane.

### Phase 2 — NAT GLMsingle producer + 6 fits (critical path — start immediately after Phase 1)
**Create** `scripts/glmsingle_natencoding.py` (clone/adapt `glmsingle_tbencoding.py`):
- `build_chunk_conditions()` per D7; write `condition_key.csv` (col_index, movie_name, chunk_idx, n_presentations) + `chunk_info.csv` (session, run, run_idx, movie_name, chunk_idx, onset, onset_vol, col_index).
- Per-run design matrices `(n_vols, ~3280)`; same GLMsingle options as TB (`wantlibrary=1, wantglmdenoise=1, wantfracridge=1`, per-session `sessionindicator`), `stimdur=4.5`.
- **Memory control (required)**: NAT is 14,077 vols × ~3,280 conds vs TB's 8,820 × 2,562 that needed 200 GB. Default: pass **masked 2D data** `(V_brain, T)` (union of run brain masks, ~220k voxels) → target ≤150 GB; save `brain_mask_index.npy` + mask NIfTI beside outputs; downstream loader handles both 4D and masked-2D layouts. Fallback: high-mem node at 500 GB for full-grid.

**Create** `scripts/glmsingle_natencoding.sbatch`: array of 6 (sub × pipeline), 8 CPUs, `--mem=180G`, `--time=36:00:00`, no `--mail-type`.
**Verify**: chunk_info rows = Σ chunks over presentations; 115 conditions with n_presentations=10; betas trailing dim matches; identical chunk_info across pipelines; mean split-half reliability of repeated-movie chunk betas in EVC > 0 (via `compute_voxel_reliability` adaptation).
**Known risks (disclose in report, no blocker)**: adjacent-chunk HRF overlap → correlated neighboring betas (fracridge shrinks; structure identical across cells and pipelines so contrasts are fair; TB's triplet has the same 4.5 s SOA); onset jitter ≤0.75 s identical across pipelines.

### Phase 3 — rawTR extraction (parallel with Phase 2)
**Create** `scripts/pattern_similarity/extract_rawtr.py`: per (pipeline, sub, ses, run) — load BOLD + brain mask, slice to ROI voxels **before** regression, Model-8 voxelwise residualization (`confound_matrix_for_run` + `regress_confounds`), per-voxel z-score, then:
- TB: per image trial, pattern = mean vols `onset_tr+3, +4` → per-ROI `(V_roi, 61)` + (mmmId, onset) metadata.
- NAT: per movie, per-TR patterns shifted +3 TRs → per-ROI `(V_roi, n_movie_TRs)` + (movie_name, tr_index).
- Write `derivatives/pattern_similarity/cache/rawtr/{pipeline}/{sub}/{sub}_{ses}_task-*_run-*_desc-model8_roipatterns.npz` (all 6 ROIs per file, float32).

**Create** `extract_rawtr.sbatch`: array of 6, 4 CPUs / 32 G / 4 h (62 runs each).
**Verify**: 62 npz per (sub × pipeline); TB n_trials=61; NAT TR sums match events; residual ROI-mean ~uncorrelated with confound columns on a spot-check run.

### Phase 4 — TB GLMsingle ROI cache (parallel with 2–3)
**Create** `scripts/pattern_similarity/extract_betas.py` (`--paradigm TB|NAT`): load TYPED betas once per (sub × pipeline) via `tb_eval.load_glmsingle_betas` + `load_trial_metadata`, slice to bilateral ROIs → `cache/glmsingle/{pipeline}/{sub}/..._task-TBencoding_desc-typed_roipatterns.npz` with (session, run, mmmId) index arrays.
**sbatch**: array of 6, 64 G / 2 h. **Verify**: trailing dims == trial_info rows; ROI voxel counts match Phase 3 caches.

### Phase 5 — NAT GLMsingle ROI cache (after Phase 2)
Same script, `--paradigm NAT` against `GLMSINGLE_NAT_DIRS` + `chunk_info.csv`, handling masked-2D layout via `brain_mask_index.npy`. **Verify**: trailing dim == chunk_info rows; repeated-movie chunks have exactly 10 presentations in the index.

### Phase 6 — Similarity computation (after 3+4+5)
**Create** `scripts/pattern_similarity/similarity.py` — cache-only I/O. Core `pattern_corr(A, B)`: column-matched Pearson r over mutually finite voxels, averaged over matched columns.
- **NAT within-sub same**: per subject × repeated movie × timeseries — all 45 session pairs, TR/chunk-matched (truncate to min shared length as a guard).
- **NAT within-sub diff**: per same-item pair, 10 seeded draws of another movie from the second member's run, contiguous window subsampled to same TR/chunk count.
- **TB within-sub same**: per subject × triplet mmmId — all cross-session presentation pairs (within-session pairs excluded; 42 presentations → 819 cross-session pairs/item).
- **TB within-sub diff**: 10 seeded draws of contiguous 3-trial windows of non-repeated items from the second member's run, position-matched.
- **Across-subject** (3 subject pairs): NAT all 10×10 session pairs + `session_scope=matched` diagonal subset; TB matched by mmmId over **all 6 shared items** (42 presentations each per subject; cross-session pairs only), with `context_match ∈ {triplet-triplet, single-single, mixed}` recorded per pair; different-item surrogates drawn from subject B's runs.
- Outputs: `results/similarity_summary.tsv` (RESULT_COLS, row per unit × item × cell × combo) + collapsed `results/similarity_cells.tsv`.

**sbatch**: single job, 4 CPUs / 32 G / 2 h. **Verify**: n_pairs match closed-form counts (45 NAT within; 819×items TB within; 100×3 NAT across); all 4 cells present for every (paradigm × pipeline × timeseries × roi); sanity: EVC NAT within_same > within_diff.

### Phase 7 — Report (after 6)
**Create** `scripts/pattern_similarity/report.py` following `nordic_benchmark/report.py`/`report_html.py`: 4-cell bar figures per ROI × paradigm × pipeline × timeseries, paired Wilcoxon (same vs diff; within vs across; nordic vs original deltas), self-contained HTML at `derivatives/pattern_similarity/report/index.html`.
**Verify**: report renders; stats rows = 6 ROIs × 8 stream combos × contrasts.

## Ordering / parallelism
1. Phase 0 + 1 first (login node, same day).
2. Phases 2, 3, 4 run in parallel after Phase 1. **Phase 2 is the critical path** (~18–30 h × 6 array tasks).
3. Phase 5 after 2; Phase 6 after 3+4+5; Phase 7 last.

## Risk register
1. NAT GLMsingle memory — masked-2D default (≤~180 G), high-mem 500 G fallback; 128 G is known-insufficient for the smaller TB problem.
2. Adjacent-chunk collinearity/bleed — fracridge shrinkage; identical across cells/pipelines; disclosed.
3. ~~Triplet identity across subjects~~ — RESOLVED: intentional per-subject shuffle of the fixed 6-item shared pool; across-subject cell uses all 6 items with `context_match` recorded.
4. Across-subject alignment is MNI-only (no hyperalignment) — depresses across-subject r uniformly; disclosed; hyperalignment is a listed roadmap extension.
5. ses-28 has 3 movies/run — Task 0 gates the same-run different-item sampler (needs ≥2 movies/run; 3 satisfies it).
6. Across-subject context mismatch (an item can be triplet-context in one subject, single-context in another) — handled by the `context_match` column; report contrasts can subset to matched-context pairs.

## Optional extensions (not core)
- Super-repeat single items (pairId 75–77, `enCon==2`, 42 presentations each) as additional **within-subject** same-items (they are already core for the across-subject cell).
- Reliability-thresholded voxel selection via `glmsingle_qc.compute_voxel_reliability`.
- Lateralized (L/R) ROI variant via `--split-hemi`.
