#!/usr/bin/env python3
"""Extract COCO metadata (captions, object categories) for the 1,000 NSD shared images.

Reads the NSD stimulus info CSV to get COCO IDs, then extracts matching
annotations from the COCO captions and instances JSON files.

Produces two output files in stimuli/shared1000/:
  - coco_annotations.csv: one row per image with captions and object categories
  - coco_captions.csv: one row per caption (5 per image, 5,000 rows total)
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

STIMULI_DIR = Path("/projects/hulacon/shared/mmmdata/stimuli/shared1000")
NSD_INFO = STIMULI_DIR / "nsd_stim_info.csv"
ANNOTATIONS_DIR = Path("/tmp/annotations")
CAPTIONS_JSON = ANNOTATIONS_DIR / "captions_train2017.json"
INSTANCES_JSON = ANNOTATIONS_DIR / "instances_train2017.json"


def load_shared_coco_ids():
    """Load the 1,000 COCO IDs for shared1000 images from the NSD metadata."""
    coco_id_to_nsd = {}
    with open(NSD_INFO) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("shared1000") == "True":
                coco_id = int(row["cocoId"])
                nsd_id = int(row["nsdId"])
                coco_id_to_nsd[coco_id] = nsd_id
    return coco_id_to_nsd


def extract_captions(coco_ids: set):
    """Extract captions for the target COCO IDs."""
    print(f"Loading captions from {CAPTIONS_JSON}...")
    with open(CAPTIONS_JSON) as f:
        data = json.load(f)

    captions_by_image = defaultdict(list)
    for ann in data["annotations"]:
        if ann["image_id"] in coco_ids:
            captions_by_image[ann["image_id"]].append(ann["caption"].strip())

    print(f"  Found captions for {len(captions_by_image)} / {len(coco_ids)} images")
    return captions_by_image


def extract_instances(coco_ids: set):
    """Extract object instance annotations for the target COCO IDs."""
    print(f"Loading instances from {INSTANCES_JSON}...")
    with open(INSTANCES_JSON) as f:
        data = json.load(f)

    # Build category ID -> name mapping
    cat_map = {c["id"]: c["name"] for c in data["categories"]}
    supercat_map = {c["id"]: c["supercategory"] for c in data["categories"]}

    # Collect unique categories per image
    objects_by_image = defaultdict(lambda: defaultdict(int))
    for ann in data["annotations"]:
        if ann["image_id"] in coco_ids:
            cat_name = cat_map[ann["category_id"]]
            objects_by_image[ann["image_id"]][cat_name] += 1

    # Also collect supercategories
    supercats_by_image = defaultdict(set)
    for ann in data["annotations"]:
        if ann["image_id"] in coco_ids:
            supercats_by_image[ann["image_id"]].add(
                supercat_map[ann["category_id"]]
            )

    print(f"  Found object annotations for {len(objects_by_image)} / {len(coco_ids)} images")
    return objects_by_image, supercats_by_image


def main():
    # Load target COCO IDs
    coco_id_to_nsd = load_shared_coco_ids()
    coco_ids = set(coco_id_to_nsd.keys())
    print(f"Loaded {len(coco_ids)} shared1000 COCO IDs")

    # Extract annotations
    captions_by_image = extract_captions(coco_ids)
    objects_by_image, supercats_by_image = extract_instances(coco_ids)

    # --- Write per-image summary CSV ---
    summary_path = STIMULI_DIR / "coco_annotations.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "nsdId",
            "cocoId",
            "caption_1",
            "caption_2",
            "caption_3",
            "caption_4",
            "caption_5",
            "object_categories",
            "object_counts",
            "supercategories",
            "n_object_instances",
            "n_unique_categories",
        ])

        for coco_id in sorted(coco_ids):
            nsd_id = coco_id_to_nsd[coco_id]
            caps = captions_by_image.get(coco_id, [])
            # Pad to 5 captions
            caps_padded = (caps + [""] * 5)[:5]

            objects = objects_by_image.get(coco_id, {})
            # Sort categories alphabetically
            sorted_cats = sorted(objects.keys())
            cat_str = "; ".join(sorted_cats)
            count_str = "; ".join(f"{c}:{objects[c]}" for c in sorted_cats)
            n_instances = sum(objects.values())
            n_categories = len(sorted_cats)

            supercats = supercats_by_image.get(coco_id, set())
            supercat_str = "; ".join(sorted(supercats))

            writer.writerow([
                nsd_id,
                coco_id,
                *caps_padded,
                cat_str,
                count_str,
                supercat_str,
                n_instances,
                n_categories,
            ])

    print(f"\nWrote per-image summary: {summary_path}")

    # --- Write per-caption CSV ---
    captions_path = STIMULI_DIR / "coco_captions.csv"
    with open(captions_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["nsdId", "cocoId", "caption_index", "caption"])

        for coco_id in sorted(coco_ids):
            nsd_id = coco_id_to_nsd[coco_id]
            caps = captions_by_image.get(coco_id, [])
            for i, cap in enumerate(caps, 1):
                writer.writerow([nsd_id, coco_id, i, cap])

    print(f"Wrote per-caption file: {captions_path}")

    # --- Summary stats ---
    total_captions = sum(len(v) for v in captions_by_image.values())
    total_instances = sum(
        sum(v.values()) for v in objects_by_image.values()
    )
    all_cats = set()
    for cats in objects_by_image.values():
        all_cats.update(cats.keys())

    print(f"\n--- Summary ---")
    print(f"Images:              {len(coco_ids)}")
    print(f"Total captions:      {total_captions}")
    print(f"Total obj instances:  {total_instances}")
    print(f"Unique obj categories: {len(all_cats)}")
    print(f"Categories: {', '.join(sorted(all_cats))}")


if __name__ == "__main__":
    main()
