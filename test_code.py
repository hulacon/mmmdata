from src.python.core import summarize_bids_dataset, load_config

# Quick summary (loads config automatically)
summary = summarize_bids_dataset()

# Access the summary data
print(f"Found {summary['n_subjects']} subjects")
print(f"Tasks: {summary['tasks']}")

# Use the BIDSLayout object for further queries
layout = summary['layout']
func_files = layout.get(subject='01', datatype='func', suffix='bold')
