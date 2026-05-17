from pathlib import Path
import json
import sys

METRICS_PATH = Path("app/metrics.json")
MIN_ACCURACY = 0.65
MIN_ROC_AUC = 0.65

def main() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError("Metrics file not found. Run train.py first.")

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    accuracy = metrics.get("accuracy", 0)
    roc_auc = metrics.get("roc_auc", 0)

    print(json.dumps(metrics, indent=2))

    if accuracy < MIN_ACCURACY or roc_auc < MIN_ROC_AUC:
        print(f"Model rejected. Accuracy={accuracy}, ROC_AUC={roc_auc}.")
        sys.exit(1)

    print("Model accepted.")

if __name__ == "__main__":
    main()
