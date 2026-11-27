from __future__ import annotations

import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def _slugify(text: str) -> str:
    """Převede název modelu na bezpečný řetězec pro název souboru."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _ensure_parent_dir(file_path: str) -> None:
    """Bezpečně vytvoří nadřazenou složku, pokud je v cestě uvedena."""
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    output_path: str,
) -> None:
    """Vykreslí a uloží matici záměn s přesnými popisky TN, FP, FN, TP."""
    _ensure_parent_dir(output_path)

    tn, fp, fn, tp = cm.ravel()
    annot_labels = np.array([
        [f"TN\n{tn}", f"FP\n{fp}"],
        [f"FN\n{fn}", f"TP\n{tp}"],
    ])

    labels = ["Nespokojený / Neutrální (0)", "Spokojený (1)"]
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(
        cm,
        annot=annot_labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 12, "weight": "bold"},
    )
    plt.title(f"Matice záměn – {model_name}", fontsize=13, pad=12)
    plt.xlabel("Predikovaná třída", fontsize=11)
    plt.ylabel("Skutečná třída", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  [INFO] Matice záměn uložena do: {output_path}")


def plot_feature_importance(
    importances: pd.Series,
    model_name: str,
    output_path: str,
    top_n: int = 10,
) -> None:
    """Vykreslí a uloží graf nejdůležitějších příznaků pro stromové modely."""
    _ensure_parent_dir(output_path)

    top_features = importances.head(top_n)[::-1]

    plt.figure(figsize=(9, 6))
    top_features.plot(kind="barh", color="#2b6cb0")
    plt.title(f"Top {len(top_features)} nejdůležitějších příznaků – {model_name}", fontsize=13, pad=12)
    plt.xlabel("Relativní důležitost", fontsize=11)
    plt.ylabel("Příznak", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  [INFO] Graf důležitosti příznaků uložen do: {output_path}")


def evaluate_model(
    model,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    save_plots: bool = True,
    output_dir: str = "grafy",
    top_n_features: int = 10,
) -> dict:
    """Natrénuje model, změří časy fit/predict, spočítá metriky a uloží grafy."""
    print(f"\n{'-' * 60}")
    print(f"[MODEL] Trénování a testování: {model_name}")

    # 1. Trénování (fit)
    start_fit = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start_fit

    # 2. Predikce (predict)
    start_pred = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - start_pred

    # 3. Metriky a matice záměn (0 = neutral/dissatisfied, 1 = satisfied)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print(f"  Accuracy:  {accuracy * 100:.2f} %")
    print(f"  Precision: {precision * 100:.2f} %")
    print(f"  Recall:    {recall * 100:.2f} %")
    print(f"  F1-score:  {f1 * 100:.2f} %")
    print(f"  Matice záměn -> TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
    print(f"  Čas fit:     {fit_time:.4f} s")
    print(f"  Čas predict: {predict_time:.4f} s")

    # 4. Důležitost příznaků (pouze stromové modely)
    feature_importances = None
    if hasattr(model, "feature_importances_") and isinstance(X_train, pd.DataFrame):
        feature_importances = pd.Series(
            model.feature_importances_,
            index=X_train.columns,
        ).sort_values(ascending=False)

        print("  Top 5 nejvlivnějších faktorů spokojenosti:")
        for feat, val in feature_importances.head(5).items():
            print(f"    - {feat}: {val:.4f}")

    # 5. Uložení grafů pro poster
    if save_plots:
        slug = _slugify(model_name)
        cm_path = os.path.join(output_dir, f"cm_{slug}.png")
        plot_confusion_matrix(cm, model_name, cm_path)

        if feature_importances is not None:
            fi_path = os.path.join(output_dir, f"importance_{slug}.png")
            plot_feature_importance(
                feature_importances,
                model_name,
                fi_path,
                top_n=top_n_features,
            )

    return {
        "Model": model_name,
        "Accuracy (%)": round(accuracy * 100, 2),
        "Precision (%)": round(precision * 100, 2),
        "Recall (%)": round(recall * 100, 2),
        "F1-score (%)": round(f1 * 100, 2),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "Fit čas (s)": round(fit_time, 4),
        "Predict čas (s)": round(predict_time, 4),
        "Celkový čas (s)": round(fit_time + predict_time, 4),
        "feature_importances": feature_importances,
    }


def run_decision_tree(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    max_depth: int | None = 12,
    random_state: int = 42,
    save_plots: bool = True,
    output_dir: str = "grafy",
) -> dict:
    """1. model: Rozhodovací strom (DecisionTreeClassifier)."""
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state,
    )
    return evaluate_model(
        model=model,
        model_name="Decision Tree",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        save_plots=save_plots,
        output_dir=output_dir,
    )


def run_random_forest(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    n_estimators: int = 100,
    max_depth: int | None = 20,
    random_state: int = 42,
    n_jobs: int = -1,
    save_plots: bool = True,
    output_dir: str = "grafy",
) -> dict:
    """2. model: Náhodný les (RandomForestClassifier)."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return evaluate_model(
        model=model,
        model_name="Random Forest",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        save_plots=save_plots,
        output_dir=output_dir,
    )


def run_svm(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: str = "scale",
    cache_size: float = 1000,
    random_state: int = 42,
    save_plots: bool = True,
    output_dir: str = "grafy",
) -> dict:
    """3. model: SVM s RBF jádrem (očekává standardizovaná data z data_prep.py)."""
    model = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        cache_size=cache_size,
        random_state=random_state,
    )
    return evaluate_model(
        model=model,
        model_name="SVM (RBF)",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        save_plots=save_plots,
        output_dir=output_dir,
    )


def run_all_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 42,
    save_plots: bool = True,
    output_dir: str = "grafy",
) -> pd.DataFrame:
    """Spustí všechny tři vybrané modely a vrátí souhrnnou srovnávací tabulku."""
    results = [
        run_decision_tree(
            X_train, X_test, y_train, y_test,
            random_state=random_state,
            save_plots=save_plots,
            output_dir=output_dir,
        ),
        run_random_forest(
            X_train, X_test, y_train, y_test,
            random_state=random_state,
            save_plots=save_plots,
            output_dir=output_dir,
        ),
        run_svm(
            X_train, X_test, y_train, y_test,
            random_state=random_state,
            save_plots=save_plots,
            output_dir=output_dir,
        ),
    ]

    summary_cols = [
        "Model",
        "Accuracy (%)",
        "Precision (%)",
        "Recall (%)",
        "F1-score (%)",
        "TP",
        "TN",
        "FP",
        "FN",
        "Fit čas (s)",
        "Predict čas (s)",
        "Celkový čas (s)",
    ]
    return pd.DataFrame(results)[summary_cols]


if __name__ == "__main__":
    from data_prep import prepare_data

    X_tr, X_te, y_tr, y_te = prepare_data(
        train_path="data/train.csv",
        test_path="data/test.csv",
        scale=True,
        sample_size=10000,
        random_state=42,
        save_corr_plot=False,
    )
    df_summary = run_all_models(X_tr, X_te, y_tr, y_te, random_state=42, save_plots=True)
    print("\n" + df_summary.to_string(index=False))