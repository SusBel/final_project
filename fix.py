"""
fix_models.py  —  Run this ONCE to patch your .keras model files.

The .keras format is just a ZIP file containing a config.json.
This script unzips each model, patches the incompatible JSON keys,
and rezips it — permanently fixing the version mismatch without
needing to retrain anything.

Run:
    python fix_models.py
"""

import os
import json
import zipfile
import shutil
import tempfile

MODELS_TO_FIX = [
    "intent_model.keras",
    "emotion_model_optimized.keras",
]


def patch_config(obj):
    """
    Recursively walk the config dict and fix two known issues:

    1. InputLayer:  'batch_shape' → 'input_shape'  (strip the batch dim)
    2. DTypePolicy: {'module': 'keras', 'class_name': 'DTypePolicy', ...}
                    → just the plain string  e.g. 'float32'
    """
    if isinstance(obj, dict):
        # Fix 1: batch_shape in InputLayer config
        if "batch_shape" in obj and "class_name" not in obj:
            batch_shape = obj.pop("batch_shape")
            obj["input_shape"] = batch_shape[1:]   # drop the batch dim

        # Fix 2: DTypePolicy stored as a nested dict → flatten to string
        if (obj.get("class_name") == "DTypePolicy"
                and "config" in obj
                and "name" in obj["config"]):
            return obj["config"]["name"]           # return plain string

        return {k: patch_config(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [patch_config(i) for i in obj]

    return obj


def fix_model(model_path: str):
    if not os.path.exists(model_path):
        print(f"  SKIP  {model_path}  (file not found)")
        return

    backup_path = model_path + ".bak"
    if not os.path.exists(backup_path):
        shutil.copy2(model_path, backup_path)
        print(f"  Backup saved → {backup_path}")

    tmp_dir = tempfile.mkdtemp()
    try:
        # 1. Unzip
        with zipfile.ZipFile(model_path, "r") as zf:
            zf.extractall(tmp_dir)
            names = zf.namelist()

        # 2. Patch config.json (may be at root or inside a subfolder)
        config_candidates = [
            os.path.join(tmp_dir, "config.json"),
            os.path.join(tmp_dir, "model.json"),
        ]
        # Also search one level deep
        for fname in os.listdir(tmp_dir):
            fpath = os.path.join(tmp_dir, fname, "config.json")
            if os.path.exists(fpath):
                config_candidates.append(fpath)

        patched_any = False
        for cfg_path in config_candidates:
            if not os.path.exists(cfg_path):
                continue
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            patched = patch_config(cfg)

            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(patched, f, indent=2)

            print(f"  Patched  {cfg_path}")
            patched_any = True

        if not patched_any:
            print(f"  WARNING: no config.json found inside {model_path}")
            print(f"  Files inside zip: {names}")
            return

        # 3. Rezip back into the original path
        tmp_zip = model_path + ".tmp"
        with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(tmp_dir):
                for file in files:
                    abs_path = os.path.join(root, file)
                    arc_name = os.path.relpath(abs_path, tmp_dir)
                    zf.write(abs_path, arc_name)

        os.replace(tmp_zip, model_path)
        print(f"  Done    {model_path}  ✓\n")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    print("=" * 50)
    print("  Keras model compatibility patcher")
    print("=" * 50 + "\n")

    for model in MODELS_TO_FIX:
        print(f"Processing: {model}")
        fix_model(model)

    print("All done. Now run:  python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000")