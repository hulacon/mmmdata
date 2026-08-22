# Reviewed dcm2bids configs (duckbrain layout) — sub-06/07

The reviewed per-session dcm2bids configs for sub-06 (9 sessions) and sub-07
(5), preserved 2026-08-21 out of the `mmmduck` piloting tree before the
re-preprocessing campaign wipes it. All 14 sha256-verified against source at
copy time — `sha256sum -c MANIFEST.sha256`.

## Why these are not in `dcm2bids_overrides/`

That directory is mmmdata's own convention (`sub-XX/ses-YY_conf.json`, read by
`scripts/run_dcm2bids.py` and `src/python/dcm2bids_config/cli.py`). These use
**duckbrain's** layout instead — `sub-XX/ses-YY/dcm2bids_config.json` — because
duckbrain resolves them from a configured root. Point it here:

```toml
[paths]
dcm2bids_config_dir = "<this repo>/config/dcm2bids_reviewed"
```

Added in `duckbrain@47035aa`. Empty/unset means `sourcedata_dir`, the old
behavior. Reads fall back to `sourcedata_dir` when this tree has no entry, so
setting it never hides a config saved beside the DICOMs.

## Why they exist at all

Per duckbrain's `core/series_skip.py`, a saved config **is** the record of a
reviewed per-session skip decision — the only non-reproducible thing in
`mmmduck`. Auto-generation can recreate a config; it cannot recreate a review.

Without one, duckbrain's `_converted_status` falls back to a presence test in
which a single non-empty NIfTI marks a session COMPLETE. With one it compares
per-datatype NIfTI counts (sub-06/ses-06 expects `{'func': 18, 'fmap': 4}`),
which is what the campaign's per-run coverage criterion needs.

## These are NOT ready to use as-is

They predate two campaign decisions, so their entities are stale:

- **D2 relabel** — task labels must be `TB*` / `floc` / `prf`.
- **`sub-006` -> `sub-06`** — participant id fix.

Copy-then-edit, never copy-blind.

Full record: `mmmdata-agents/docs/workbench/reprocessing-campaign/log.md`
(2026-08-21).
