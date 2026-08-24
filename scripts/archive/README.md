# Archived scripts

Superseded scripts, kept because they are the provenance record for outputs
that still exist or are cited in older analyses. Nothing here is maintained,
and nothing here should be run. Each entry names its successor.

## The pre-0.6.0 viz2psy pipeline — archived 2026-08-23

`run_viz2psy.py`, `submit_viz2psy.sh`, `slurm_viz2psy_images.sh`,
`slurm_viz2psy_movies.sh`, `slurm_viz2psy_cues.sh`

**Successor: `scripts/stimfeat_campaign.py`.**

These ran viz2psy 0.5.0 (February 2026) over images, movie frames, and cue
images, writing wide CSVs to:

```
stimuli/shared1000/viz2psy_scores.csv
stimuli/shared1000/viz2psy_scores_dashboard.html
stimuli/movies/viz2psy_scores/<Movie_Name>_scores.csv
stimuli/movies/viz2psy_cue_scores.csv
```

Three things retired them:

1. **The column convention changed.** viz2psy 0.6.0 renamed columns to the
   Contract B §4.1 form (`resmem_memorability`, not `memorability`;
   fixed-width `clip_000`, not `clip_0`). Output from these scripts cannot be
   joined to anything current without a translation nobody maintains.
2. **They only cover viz2psy.** The §4.2 campaign runs aud2psy and word2psy
   too, over movie soundtracks, word audio, machine captions, human captions,
   transcripts, and scene annotations — none of which these reach.
3. **They are not resumable.** `stimfeat_campaign.py` treats the sidecar as
   the done-marker, so it resumes at model granularity with no state file.

Their output paths were **deleted 2026-08-23**, so these scripts now write to
directories that no longer exist. That is intended, not a bug to fix: read
`derivatives/stimuli_features/` instead, through the psytwill aggregates.

To re-extract anything they produced:

```bash
stimfeat_campaign.py plan --set shared1000       # what would run
stimfeat_campaign.py run --set shared1000 --source image --model resmem
```
