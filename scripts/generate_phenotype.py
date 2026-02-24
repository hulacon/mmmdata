#!/usr/bin/env python3
"""Generate BIDS phenotype files and participants.tsv from source Excel data.

Reads:
  - Demographics Form.xlsx → participants.tsv + phenotype/demographics.tsv
  - VVIQ.xlsx → phenotype/vviq.tsv
  - mmm_scanlog.xlsx (Final Debriefing Form tab) → phenotype/final_debriefing.tsv

All outputs go to the BIDS root. Each TSV is accompanied by a JSON sidecar.

Usage:
    python generate_phenotype.py
"""

import json
from pathlib import Path

import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────

BIDS_ROOT = Path("/gpfs/projects/hulacon/shared/mmmdata")
SCAN_LOGS = BIDS_ROOT / "sourcedata/shared/scan_logs"
SCANLOG = SCAN_LOGS / "mmm_scanlog.xlsx"
DEMOGRAPHICS = SCAN_LOGS / "Demographics Form.xlsx"
VVIQ_FILE = SCAN_LOGS / "VVIQ.xlsx"
PHENOTYPE_DIR = BIDS_ROOT / "phenotype"

# ── Subject mappings ─────────────────────────────────────────────────────────

# Demographics Form uses enrollment order (not mmm_ IDs).
# Confirmed via date cross-reference against BySession scan dates:
#   Demog 2 (2024-02-27) = mmm_03 ses-01, Demog 4 (2024-07-12) = mmm_04 ses-02,
#   Demog 5 (2024-07-25) = mmm_05 ses-01
DEMOG_TO_BIDS = {2: "sub-03", 4: "sub-04", 5: "sub-05"}

# VVIQ uses subject numbers matching mmm_ IDs directly
VVIQ_TO_BIDS = {3: "sub-03", 4: "sub-04", 5: "sub-05"}

# Final Debriefing uses mmm_ IDs
DEBRIEF_TO_BIDS = {"mmm_03": "sub-03", "mmm_04": "sub-04", "mmm_05": "sub-05"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _write_tsv_json(df: pd.DataFrame, tsv_path: Path, sidecar: dict) -> None:
    """Write a TSV and its companion JSON sidecar."""
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tsv_path, sep="\t", index=False)
    json_path = tsv_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(sidecar, f, indent=4)
        f.write("\n")
    print(f"Wrote {tsv_path} ({len(df)} rows) + {json_path.name}")


def _clean_str(v) -> str:
    """Convert a value to a clean string, replacing empty/None with n/a."""
    if pd.isna(v) or v is None:
        return "n/a"
    s = str(v).strip()
    if s in ("", "None", "nan"):
        return "n/a"
    return s


# ── Demographics → participants.tsv (all intake data in one place) ────────────

def generate_participants() -> None:
    """Generate participants.tsv with full demographics from intake form."""
    df = pd.read_excel(DEMOGRAPHICS)
    df = df[df["subject"].isin(DEMOG_TO_BIDS)].copy()
    df["participant_id"] = df["subject"].map(DEMOG_TO_BIDS)
    df = df.sort_values("participant_id").reset_index(drop=True)

    participants = pd.DataFrame()
    participants["participant_id"] = df["participant_id"]
    participants["age"] = df["age"].astype(int)
    participants["sex"] = df["sex"]
    participants["handedness"] = df["handedness"]
    participants["race"] = df["race"].apply(_clean_str)
    participants["hispanic_origin"] = df["are you of hispanic origin, regardless of race?"].apply(_clean_str)
    participants["english_first_language"] = df["is english your first language?"].apply(_clean_str)
    participants["vision"] = df["do you have normal or corrected t onromal vision?"].apply(_clean_str)
    participants["hearing"] = df["do you have normal hearing?"].apply(_clean_str)
    participants["colorblind"] = df["are you colorblind?"].apply(_clean_str)
    participants["education_level"] = df["highest grade or year of school you completed?"].apply(_clean_str)
    participants["education_years"] = df["number of years of education completed:"].apply(
        lambda v: int(v) if pd.notna(v) and v == int(v) else _clean_str(v)
    )
    participants["psychiatric_history"] = df[
        "do you have current or past history of primary psychiatric disorder"
    ].apply(_clean_str)
    participants["current_medications"] = df["are you currently taking any medications?"].apply(_clean_str)
    participants["medication_list"] = df["if yes, please list"].apply(_clean_str)
    participants["sleep_hours_enrollment"] = df["how many hours did you sleep last night"].apply(_clean_str)
    participants["sleep_hours_average_enrollment"] = df[
        "in the past month how many hours of sleep did you average per night?"
    ].apply(lambda v: int(v) if pd.notna(v) and isinstance(v, (int, float)) and v == int(v) else _clean_str(v))

    sidecar = {
        "age": {"Description": "Age at enrollment", "Units": "years"},
        "sex": {
            "Description": "Biological sex as reported by participant",
            "Levels": {"M": "male", "F": "female"},
        },
        "handedness": {
            "Description": "Self-reported handedness",
            "Levels": {"right": "right-handed", "left": "left-handed", "ambidextrous": "ambidextrous"},
        },
        "race": {"Description": "Self-reported race"},
        "hispanic_origin": {"Description": "Whether participant is of Hispanic origin, regardless of race"},
        "english_first_language": {"Description": "Whether English is the participant's first language"},
        "vision": {
            "Description": "Self-reported visual acuity",
            "Levels": {"normal": "Normal vision", "corrected": "Corrected-to-normal vision"},
        },
        "hearing": {"Description": "Whether participant has normal hearing"},
        "colorblind": {"Description": "Whether participant is colorblind"},
        "education_level": {"Description": "Highest grade or year of school completed"},
        "education_years": {"Description": "Total number of years of education completed", "Units": "years"},
        "psychiatric_history": {
            "Description": "Whether participant has current or past history of primary psychiatric disorder"
        },
        "current_medications": {"Description": "Whether participant was taking medications at enrollment"},
        "medication_list": {"Description": "List of medications if taking any at enrollment"},
        "sleep_hours_enrollment": {
            "Description": "Hours of sleep the night before the enrollment session",
            "Units": "hours",
        },
        "sleep_hours_average_enrollment": {
            "Description": "Self-reported average hours of sleep per night in the past month at enrollment",
            "Units": "hours",
        },
    }
    _write_tsv_json(participants, BIDS_ROOT / "participants.tsv", sidecar)


# ── VVIQ → phenotype/vviq.tsv ───────────────────────────────────────────────

def generate_vviq() -> None:
    """Generate phenotype/vviq.tsv from VVIQ.xlsx."""
    df = pd.read_excel(VVIQ_FILE)
    df = df[df["subject"].isin(VVIQ_TO_BIDS)].copy()
    df["participant_id"] = df["subject"].map(VVIQ_TO_BIDS)
    df = df.sort_values("participant_id").reset_index(drop=True)

    # Extract the 16 VVIQ item scores (columns 2-17, skipping scenario headers)
    # The Excel has: subject, scenario1_header, item1, item2, item3, item4,
    #                scenario2_header, item5, item6, item7, item8, ...
    # We need to pick only the numeric item columns (not the scenario headers)
    item_cols = [c for c in df.columns if c != "subject" and c != "participant_id"]
    score_cols = []
    for c in item_cols:
        # Score columns have numeric data; scenario headers are text/None
        vals = df[c].dropna()
        if len(vals) > 0 and all(isinstance(v, (int, float)) for v in vals):
            score_cols.append(c)

    vviq = pd.DataFrame()
    vviq["participant_id"] = df["participant_id"]

    # Rename score columns to vviq_01 through vviq_16
    item_descriptions = [
        "Exact contour of face, head, shoulders, and body",
        "Characteristic poses of head, attitudes of body",
        "Precise carriage, length of step in walking",
        "Different colors worn in familiar clothes",
        "Sun rising above horizon into hazy sky",
        "Sky clears and surrounds sun with blueness",
        "Storm blows up with flashes of lightning",
        "A rainbow appears",
        "Overall appearance of shop from opposite side of road",
        "Window display including colors, shapes, details of items",
        "Color, shape, and details of the door near entrance",
        "Enter shop, go to counter, assistant serves you, money changes hands",
        "Contours of the landscape",
        "Color and shape of the trees",
        "Color and shape of the lake",
        "Strong wind blows on trees and lake causing waves",
    ]

    scenario_names = [
        "Familiar person",
        "Rising sun",
        "Country shop",
        "Country scene",
    ]

    for i, col in enumerate(score_cols[:16]):
        bids_name = f"vviq_{i+1:02d}"
        vviq[bids_name] = df[col].astype(int)

    # Compute total
    score_bids = [f"vviq_{i+1:02d}" for i in range(len(score_cols[:16]))]
    vviq["vviq_total"] = vviq[score_bids].sum(axis=1).astype(int)

    # Sidecar
    sidecar: dict = {
        "participant_id": {"Description": "BIDS participant identifier"},
        "MeasurementToolName": "Vividness of Visual Imagery Questionnaire (VVIQ)",
        "MeasurementToolDescription": (
            "16-item self-report measure of the vividness of visual imagery. "
            "Four scenarios with 4 items each. Rating scale: 1 = perfectly clear "
            "and vivid as normal vision, 5 = no image at all."
        ),
    }

    for i in range(min(16, len(score_cols))):
        bids_name = f"vviq_{i+1:02d}"
        scenario_idx = i // 4
        item_in_scenario = (i % 4) + 1
        sidecar[bids_name] = {
            "Description": item_descriptions[i] if i < len(item_descriptions) else f"VVIQ item {i+1}",
            "Levels": {
                "1": "Perfectly clear and as vivid as normal vision",
                "2": "Clear and reasonably vivid",
                "3": "Moderately clear and vivid",
                "4": "Vague and dim",
                "5": "No image at all, you only know that you are thinking of the object",
            },
            "Scenario": (
                f"{scenario_names[scenario_idx]} (items {scenario_idx*4+1}-{scenario_idx*4+4})"
                if scenario_idx < len(scenario_names)
                else ""
            ),
        }

    sidecar["vviq_total"] = {
        "Description": "Sum of all 16 VVIQ items (range: 16-80). Lower scores indicate more vivid imagery.",
    }

    _write_tsv_json(vviq, PHENOTYPE_DIR / "vviq.tsv", sidecar)


# ── Final Debriefing → phenotype/final_debriefing.tsv ────────────────────────

def generate_final_debriefing() -> None:
    """Generate phenotype/final_debriefing.tsv from scanlog."""
    df = pd.read_excel(SCANLOG, sheet_name="Final Debriefing Form")

    # Drop empty rows
    df = df.dropna(subset=[df.columns[0]]).copy()
    df = df[df.iloc[:, 0].isin(DEBRIEF_TO_BIDS)].copy()
    df["participant_id"] = df.iloc[:, 0].map(DEBRIEF_TO_BIDS)
    df = df.sort_values("participant_id").reset_index(drop=True)

    # Map columns
    headers = list(df.columns)

    def _find_col(keyword: str) -> str | None:
        for h in headers:
            if h and keyword.lower() in str(h).lower():
                return h
        return None

    debrief = pd.DataFrame()
    debrief["participant_id"] = df["participant_id"]
    debrief["session_id"] = "ses-30"

    col_map = [
        ("overall experience", "overall_experience"),
        ("more challenging", "more_challenging_task"),
        ("strategies to remember", "memory_strategies"),
        ("changes in your memory", "memory_changes_noticed"),
        ("thinking about the memory tasks outside", "thinking_outside_sessions"),
        ("favorite or least favorite", "favorite_least_favorite"),
        ("how many and which movies", "movies_seen_previously"),
        ("recognize any of the actors", "actors_recognized"),
        ("experience of the movies change", "repeated_movie_experience_change"),
        ("experience of recalling the movies change", "repeated_movie_recall_change"),
        ("perception of how time passed", "repeated_movie_time_perception"),
        ("notice any repetition", "image_word_repetition_noticed"),
        ("how likely would you be", "likelihood_to_repeat"),
    ]

    for keyword, bids_name in col_map:
        src = _find_col(keyword)
        if src:
            debrief[bids_name] = df[src].apply(_clean_str)
        else:
            debrief[bids_name] = "n/a"

    sidecar = {
        "participant_id": {"Description": "BIDS participant identifier"},
        "session_id": {
            "Description": "BIDS session in which the debriefing was administered (Final Cued Recall)"
        },
        "overall_experience": {
            "Description": "Overall study experience rating",
            "Levels": {"1": "Terrible", "2": "Poor", "3": "Average", "4": "Good", "5": "Great"},
        },
        "more_challenging_task": {
            "Description": "Which task type (free or cued recall) the participant found more challenging, with explanation"
        },
        "memory_strategies": {
            "Description": "Strategies used to remember information for videos or image-word pairs"
        },
        "memory_changes_noticed": {
            "Description": "Whether participant noticed changes in memory, attention, or mental clarity across the study"
        },
        "thinking_outside_sessions": {
            "Description": "Whether participant thought about memory tasks outside of study sessions"
        },
        "favorite_least_favorite": {
            "Description": "Participant's favorite and least favorite parts of the study"
        },
        "movies_seen_previously": {
            "Description": "Which movies the participant had seen before the study"
        },
        "actors_recognized": {
            "Description": "Whether participant recognized any actors in the movies"
        },
        "repeated_movie_experience_change": {
            "Description": "How the experience of watching the two repeated movies (From Dad to Son, The Bench) changed across sessions"
        },
        "repeated_movie_recall_change": {
            "Description": "How the experience of recalling the two repeated movies changed across sessions"
        },
        "repeated_movie_time_perception": {
            "Description": "Whether perception of time passing during the repeated movies changed across sessions"
        },
        "image_word_repetition_noticed": {
            "Description": "Whether participant noticed any repetition of items or sequences in image-word pairs"
        },
        "likelihood_to_repeat": {
            "Description": "On a scale of 1-10, how likely the participant would be to do this kind of experiment again (1=unlikely, 10=likely), with explanation"
        },
    }

    _write_tsv_json(debrief, PHENOTYPE_DIR / "final_debriefing.tsv", sidecar)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    PHENOTYPE_DIR.mkdir(parents=True, exist_ok=True)
    generate_participants()
    generate_vviq()
    generate_final_debriefing()
    print("\nDone. All phenotype files written.")


if __name__ == "__main__":
    main()
