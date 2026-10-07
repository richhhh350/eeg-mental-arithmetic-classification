"""Copy an explicit small evidence bundle into the repository; no raw EEG/models."""
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


root = Path(__file__).resolve().parents[1]
destination = root.parent / "results" / "v2-stage1"
destination.mkdir(parents=True, exist_ok=True)
sources = {
    "loso-stage1": ["subject_metrics.csv", "predictions.csv", "summary.json", "verification.json", "run.json", "subject_balanced_accuracy.png"],
    "prepared-stage1": ["manifest.json", "qc_waveforms.png", "qc_psd.png"],
    "legacy_validation": ["legacy_validation.json"],
}
for directory, names in sources.items():
    for name in names:
        source = root / "outputs" / directory / name
        shutil.copy2(source, destination / name)
for name in ("tests.xml", "independent_audit.json"):
    source = root / "outputs" / name
    if source.exists():
        if name == "tests.xml":
            tree = ET.parse(source)
            for element in tree.iter():
                element.attrib.pop("hostname", None)
            tree.write(destination / name, encoding="utf-8", xml_declaration=True)
        else:
            shutil.copy2(source, destination / name)
print(f"Exported stage1 evidence to {destination}")
