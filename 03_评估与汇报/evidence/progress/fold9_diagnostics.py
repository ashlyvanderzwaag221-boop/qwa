"""Reproduce the source audit and optionally compare class weights on milestone fold 9.

From the project root, run python path/to/fold9_diagnostics.py --class-weights.
The optional comparison needs the milestone feature archive; the source audit alone
only requires the per-clip predictions in the repository.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def source_breakdown(rows):
    by_source = defaultdict(Counter)
    for row in rows:
        if row["actual"] == "air_conditioner":
            by_source[row["fsID"]][row["predicted"]] += 1
    details = [
        {"fsID": key, "clips": sum(classes.values()), "as_drilling": classes["drilling"]}
        for key, classes in by_source.items()
    ]
    details.sort(key=lambda row: (-row["as_drilling"], row["fsID"]))
    return {
        "air_conditioner_clips": sum(row["clips"] for row in details),
        "air_conditioner_sources": len(details),
        "predicted_drilling": sum(row["as_drilling"] for row in details),
        "sources_contributing_errors": sum(row["as_drilling"] > 0 for row in details),
        "errors_from_two_largest_contributors": sum(row["as_drilling"] for row in details[:2]),
        "by_source": details,
    }


def compare_class_weights(project, rows):
    from sklearn.metrics import f1_score
    from threadpoolctl import threadpool_limits

    sys.path.insert(0, str(project / "02_模型训练"))
    from experiment import LABELS, development_split, load_features, make_model, scores

    data_stage = project / "01_数据处理与特征"
    data_file = data_stage / "features" / "features.npz"
    protocol_file = project / "02_模型训练" / "results" / "protocol.json"
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    if hashlib.sha256(data_file.read_bytes()).hexdigest() != protocol["feature_file_sha256"]:
        raise ValueError("Feature archive differs from the one used for the milestone.")
    features, targets, metadata = load_features(data_stage / "results/manifest.csv", data_file)
    train, val, _, _ = development_split(metadata)
    val_ids = [entry["clip_id"] for entry, flag in zip(metadata, val) if flag]
    by_id = {row["clip_id"]: row for row in rows}
    if set(val_ids) != set(by_id):
        raise ValueError("Saved validation predictions and feature rows differ.")
    truth = targets[val]
    if [int(by_id[key]["actual_id"]) for key in val_ids] != truth.tolist():
        raise ValueError("Validation targets differ from the saved predictions.")
    class_scores = {}
    params = {"kernel": "rbf", "C": 10, "gamma": "scale"}
    # Only the class_weight argument changes between these two runs.
    with threadpool_limits(limits=1):
        for weight in (None, "balanced"):
            model = make_model("rbf_svm", {**params, "class_weight": weight})
            model.fit(features[train], targets[train])
            predicted = model.predict(features[val])
            if weight is None:
                previous = [int(by_id[key]["predicted_id"]) for key in val_ids]
                if previous != predicted.tolist():
                    raise ValueError("Reference predictions differ from the milestone.")
            per_class = f1_score(truth, predicted, labels=range(len(LABELS)), average=None, zero_division=0)
            class_scores[weight or "unweighted"] = {
                **scores(truth, predicted),
                "correct": int((truth == predicted).sum()),
                "per_class_f1": dict(zip(LABELS, per_class.tolist())),
            }
    return {
        "scope": "Controlled comparison on the milestone development fold; no oversampling.",
        "parameters": params,
        "train_clips": int(train.sum()),
        "validation_clips": len(truth),
        "scores": class_scores,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--class-weights", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    project = args.project_root.resolve()
    predictions = project / "02_模型训练" / "results" / "predictions.csv"
    with predictions.open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["model"] == "rbf_svm" and row["fold"] == "9"]
    if len(rows) != 816 or len({row["clip_id"] for row in rows}) != len(rows):
        raise ValueError("Expected one milestone RBF-SVM prediction per fold-9 clip.")
    report = {"validation_fold": 9, "source_audit": source_breakdown(rows)}
    if args.class_weights:
        report["class_weight_comparison"] = compare_class_weights(project, rows)
    output = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
