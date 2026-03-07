"""Rewrite imports from flat layout to package layout across all src/deltapd/ files."""
import glob
import re

# Map old module name -> new module name (within deltapd package)
IMPORT_MAP = {
    "preprocessing": "deltapd.signal_model",
    "blind_algorithms": "deltapd.trackers",
    "data_loader": "deltapd.loader",
    "descriptor_evaluation": "deltapd.features",
    "descriptors": "deltapd.descriptors",
    "validation": "deltapd.validation",
    "roc_analysis": "deltapd.roc",
    "sensitivity_analysis": "deltapd.sensitivity",
    "baseline_comparison": "deltapd.baselines",
    "empirical_validation": "deltapd.empirical",
}

files = glob.glob("src/deltapd/*.py")
for fpath in files:
    text = open(fpath, "r", encoding="utf-8").read()
    changed = False
    for old, new in IMPORT_MAP.items():
        # from old_module import ...
        pattern = rf"from {old} import"
        replacement = f"from {new} import"
        if re.search(pattern, text):
            text = re.sub(pattern, replacement, text)
            changed = True
        # import old_module
        pattern2 = rf"^import {old}$"
        replacement2 = f"import {new}"
        if re.search(pattern2, text, re.MULTILINE):
            text = re.sub(pattern2, replacement2, text, flags=re.MULTILINE)
            changed = True
    if changed:
        open(fpath, "w", encoding="utf-8", newline="\n").write(text)
        print(f"  fixed: {fpath}")

print("Done.")
