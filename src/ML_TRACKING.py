import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.features import FEATURES

# (train fraction, n_estimators) for each run
RUNS = [(0.5, 10), (0.8, 100), (0.7, 200)]


def train(runs=RUNS):
    data = pd.read_csv("dataset/phishing_domains_1000.csv")

    x = data[FEATURES]
    y = data["label"].map({1: "Fake", 0: "Genuine"})

    mlflow.set_experiment("phishing")

    # continue numbering from the models already in this experiment (v1, v2, v3 -> next is v4)
    exp_id = mlflow.get_experiment_by_name("phishing").experiment_id
    old = mlflow.search_logged_models(experiment_ids=[exp_id], output_format="list")
    last = max([int(m.name[1:]) for m in old if m.name[1:].isdigit()], default=0)

    best_f1 = 0
    results = []

    for i, (train_size, n_estimators) in enumerate(runs, start=last + 1):
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=train_size, random_state=42, stratify=y)

        with mlflow.start_run(run_name=f"v{i} ({n_estimators} trees)",
                              description=f"RandomForest with {n_estimators} trees, {train_size:.0%} train / {1 - train_size:.0%} test"):
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, pos_label="Fake")
            rec = recall_score(y_test, y_pred, pos_label="Fake")
            f1 = f1_score(y_test, y_pred, pos_label="Fake")

            mlflow.log_param("train_size", train_size)
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1", f1)
            mlflow.sklearn.log_model(model, name=f"v{i}", serialization_format="cloudpickle")

        print(f"train={train_size}  trees={n_estimators}  accuracy={acc * 100:.2f}%  "
              f"precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}")
        results.append({"version": f"v{i}", "train_size": train_size, "n_estimators": n_estimators,
                        "accuracy": round(acc, 4), "precision": round(prec, 4),
                        "recall": round(rec, 4), "f1": round(f1, 4)})

        # phishing detection cares about catching phishing, so the best model is picked by F1
        if f1 > best_f1:
            best_f1 = f1
            joblib.dump(model, "models/model.pkl")

    return results


if __name__ == "__main__":
    train()
