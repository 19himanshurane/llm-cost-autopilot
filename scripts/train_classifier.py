"""
Phase 2, Step 3: Train the classifier.

Loads data/labeled_prompts.csv, converts every prompt into numeric features
(via app/routing/features.py), splits into train/test sets, trains a
LogisticRegression classifier, reports accuracy + a confusion matrix, and
saves the trained model to data/classifier.joblib for later reuse (Phase 2
Step 4 -- the router -- will load this file rather than retraining every time).

Run from the project root:
    python -m scripts.train_classifier
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from app.routing.features import featurize_prompt, FEATURE_NAMES
from app.routing.tiers import ComplexityTier, TIER_DEFINITIONS


def load_dataset(csv_path: Path) -> tuple[list[list[float]], list[int]]:
    X: list[list[float]] = []
    y: list[int] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append(featurize_prompt(row["prompt"]))
            y.append(int(row["tier"]))
    return X, y


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    csv_path = data_dir / "labeled_prompts.csv"

    if not csv_path.exists():
        raise SystemExit(
            f"{csv_path} not found. Run `python -m scripts.generate_labeled_dataset` first."
        )

    X, y = load_dataset(csv_path)
    print(f"Loaded {len(X)} labeled examples with {len(FEATURE_NAMES)} features each.")

    # stratify=y keeps the 3 tiers proportionally represented in both the
    # train and test split -- without it, a random split could accidentally
    # put almost all Tier 3 examples into the training set and leave the
    # test set unable to fairly judge Tier 3 performance.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set: {len(X_train)} examples | Test set: {len(X_test)} examples")

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy on held-out test set: {accuracy:.1%}")

    tier_labels = [TIER_DEFINITIONS[ComplexityTier(t)].label for t in sorted(set(y))]
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=tier_labels))

    print("Confusion matrix (rows = actual tier, columns = predicted tier):")
    print(f"{'':12s}" + "".join(f"{label:>10s}" for label in tier_labels))
    cm = confusion_matrix(y_test, y_pred)
    for label, row in zip(tier_labels, cm):
        print(f"{label:12s}" + "".join(f"{val:>10d}" for val in row))

    model_path = data_dir / "classifier.joblib"
    joblib.dump(model, model_path)
    print(f"\nSaved trained model to {model_path}")

    if accuracy < 0.80:
        print(
            "\nNOTE: accuracy is below the 80% V1 target from the spec. "
            "Since this dataset is template-generated, low accuracy usually "
            "means two tiers overlap too much in the features -- check the "
            "confusion matrix above to see which tiers get confused."
        )
    else:
        print(
            "\nNOTE: because this dataset is template-generated (many "
            "prompts share very similar phrasing within a tier), accuracy "
            "here will likely look higher than it would on messier, "
            "real-world prompts. Treat this as a sanity check that the "
            "pipeline works, not a final quality guarantee."
        )


if __name__ == "__main__":
    main()