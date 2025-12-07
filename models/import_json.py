import json
from pathlib import Path

# Path to your metadata file
METADATA_FILE = "models/metadata.json"

def clean_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = {}

    for model_name, meta in data.items():
        new_entry = meta.copy()

        # -------------------------------
        # 1. Convert "filepath" → "filename"
        # -------------------------------
        fp = meta.get("filepath")
        if fp:
            filename = Path(fp).name  # keep only "gbr_one.pkl"
            new_entry["filename"] = filename
            del new_entry["filepath"]

        # -------------------------------
        # 2. Convert "training_dataset" absolute path → dataset name
        #    "C:/.../one.pkl" → "one"
        # -------------------------------
        td = meta.get("training_dataset")
        if td:
            dataset_name = Path(td).stem  # "one"
            new_entry["training_dataset"] = dataset_name

        # Save cleaned entry
        new_data[model_name] = new_entry

    # Write cleaned metadata
    with open(path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2)

    print("Metadata cleaned successfully!")


if __name__ == "__main__":
    clean_metadata(METADATA_FILE)
