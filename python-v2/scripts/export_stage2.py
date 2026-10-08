"""Export an explicit, model-free evidence bundle for completed stage2."""
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from eeg_v2.provenance import read_json


root = Path(__file__).resolve().parents[1]
runs = root / "outputs/runs-stage2"
analysis = root / "outputs/analysis-stage2"
status = read_json(runs / "batch_status.json")
if status["status"] != "complete":
    raise RuntimeError("Cannot export an incomplete stage2")
destination = root.parent / "results/v2-stage2"
destination.mkdir(parents=True, exist_ok=True)
for name in ("experiment_summary.csv", "paired_descriptive.json", "analysis.json", "s1_comparison.png", "s2_comparison.png"):
    shutil.copy2(analysis / name, destination / name)
for name in ("batch.json", "batch_status.json"):
    shutil.copy2(runs / name, destination / name)
shutil.copy2(root / "outputs/prepared-stage2/manifest.json", destination / "manifest.json")
shutil.copy2(root / "configs/stage2.json", destination / "config.json")
for spec in read_json(runs / "batch.json")["experiments"]:
    folder = destination / "experiments" / spec["id"]
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("run.json", "summary.json", "subject_metrics.csv", "predictions.csv", "verification.json"):
        shutil.copy2(runs / spec["id"] / name, folder / name)
for source in sorted(analysis.glob("pca_Subject*")):
    folder = destination / source.name
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "coordinates.csv", "subject_directions.csv", "full_dimension_descriptors.json", "pca_subjects.png", "pca_conditions.png"):
        shutil.copy2(source / name, folder / name)
tree = ET.parse(root / "outputs/stage2-tests.xml")
for element in tree.iter():
    element.attrib.pop("hostname", None)
tree.write(destination / "tests.xml", encoding="utf-8", xml_declaration=True)
audit = root / "outputs/stage2-independent-audit.json"
if audit.exists():
    shutil.copy2(audit, destination / "independent_audit.json")
print(f"Exported stage2 evidence to {destination}")
