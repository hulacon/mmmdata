---
title: Raw-to-BIDS Converters
parent: Code Documentation
nav_order: 53
has_children: true
---

# Raw-to-BIDS Converters

Modules in the `raw2bids_converters` package.

| Module | Description |
|--------|-------------|
| [behavioral_to_beh](raw2bids_converters_behavioral_to_beh) | Convert out-of-scanner behavioral CSVs into BIDS _beh.tsv files. |
| [common](raw2bids_converters_common) | Shared utilities for raw-to-BIDS behavioral data converters. |
| [edf_to_physio](raw2bids_converters_edf_to_physio) | Convert EyeLink EDF files into BIDS _recording-eye_physio.tsv.gz files. |
| [generate_inventory](raw2bids_converters_generate_inventory) | Generate file_inventory.csv from current sourcedata contents. |
| [localizer_events](raw2bids_converters_localizer_events) | Convert localizer timing CSVs into BIDS _events.tsv files. |
| [physio_dcm](raw2bids_converters_physio_dcm) | Convert Siemens PhysioLog DICOMs into BIDS _physio.tsv.gz files. |
| [psychopy_encoding](raw2bids_converters_psychopy_encoding) | Convert PsychoPy free recall encoding CSVs into BIDS _events.tsv files. |
| [psychopy_retrieval](raw2bids_converters_psychopy_retrieval) | Convert PsychoPy free recall retrieval CSVs into BIDS _events.tsv files. |
| [run_all](raw2bids_converters_run_all) | Orchestrator: read file_inventory.csv and run all converters. |
| [timed_events](raw2bids_converters_timed_events) | Convert behavioral CSV + timing CSV pairs into BIDS _events.tsv files. |
| [validate](raw2bids_converters_validate) | Validate generated cued recall BIDS events against metainformation reference. |
