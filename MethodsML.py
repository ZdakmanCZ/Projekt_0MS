import os
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
    """Převede název modelu na bezpečný název souboru."""
    return text.lower().replace(" ", "_").replace("(", "").replace(")", "")


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    output_path: str,
) -> None:
    """Vykreslí a uloží matici záměn s přesnými počty TN, FP, FN, TP."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    tn, fp, fn, tp = cm.ravel()
    annot_labels = np.array([
        [f"TN\n{tn}", f"FP\n{fp}"],
        [f"FN\n{fn}", f"TP\n{tp}"],
    ])

    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(
        cm,
        annot=annot_labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Nespokojený / Neutrální (0)", "Spokojený (1)"],
        yticklabels=["Nespokojený / Neutrální (0)", "Spokojený (1)"],
        annot_kws={"size": 12, "weight": "bold"},
    )
    plt.title(f"Matice záměn – {model_name}", fontsize=13, pad=12)
    plt.xlabel("Predikovaná třída")
    plt.ylabel("Skutečná třída")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Matice záměn uložena do: {output_path}")


def plot_feature_importance(
    importances: pd.Series,
    model_name: str,
    output_path: str,
    top_n: int = 10,
) -> None:
    """Vykreslí a uloží graf nejdůležitějších příznaků pro stromové modely."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    top_features = importances.head(top_n)[::-1]

    plt.figure(figsize=(9, 6))
    top_features.plot(kind="barh", color="#2b6cb0")
    plt.title(f"Top {top_n} nejdůležitějších příznaků – {model_name}", fontsize=13, pad=12)
    plt.xlabel("Relativní důležitost")
    plt.ylabel("Příznak")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Graf důležitosti příznaků uložen do: {output_path}")


def evaluate_model(
    model,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    save_plots: bool = True,
    output_dir: str = "grafy",
) -> dict:
    """
    Natrénuje model, samostatně změří časy fit a predict, spočítá metriky
    (Accuracy, Precision, Recall, F1, TP, TN, FP, FN) a volitelně uloží grafy.
    """
    # 1. Trénování a měření času fit
    start_fit = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start_fit

    # 2. Predikce a měření času predict
    start_pred = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - start_pred

    # 3. Výpočet metrik a matice záměn (0 = nespokojený/neutrální, 1 = spokojený)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # 4. Výpis přehledu do konzole
    print("=" * 60)
    print(f"VÝSLEDKY MODELU: {model_name}")
    print("=" * 60)
    print(f"Accuracy  (Přesnost celkem): {accuracy * 100:.2f} %")
    print(f"Precision (Přesnost třídy 1): {precision * 100:.2f} %")
    print(f"Recall    (Úplnost třídy 1):  {recall * 100:.2f} %")
    print(f"F1-score:                     {f1 * 100:.2f} %")
    print("-" * 60)
    print(f"TP (Správně spokojený):   {tp}")
    print(f"TN (Správně nespokojený): {tn}")
    print(f"FP (Falešně spokojený):   {fp}")
    print(f"FN (Falešně nespokojený): {fn}")
    print("-" * 60)
    print(f"Čas trénování (fit):     {fit_time:.4f} s")
    print(f"Čas predikce  (predict): {predict_time:.4f} s")

    # 5. Důležitost příznaků (pro Decision Tree a Random Forest)
    feature_importances = None
    if hasattr(model, "feature_importances_") and isinstance(X_train, pd.DataFrame):
        feature_importances = pd.Series(
            model.feature_importances_,
            index=X_train.columns,
        ).sort_values(ascending=False)

        print("-" * 60)
        print("Top 5 nejdůležitějších příznaků:")
        for feat, val in feature_importances.head(5).items():
            print(f"  - {feat}: {val:.4f}")

    print("=" * 60)

    # 6. Uložení grafů pro poster
    slug = _slugify(model_name)
    if save_plots:
        cm_path = os.path.join(output_dir, f"cm_{slug}.png")
        plot_confusion_matrix(cm, model_name, cm_path)

        if feature_importances is not None:
            fi_path = os.path.join(output_dir, f"feature_importance_{slug}.png")
            plot_feature_importance(feature_importances, model_name, fi_path)

    return {
        "model_name": model_name,
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
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
) -> dict:
    """Spustí a vyhodnotí rozhodovací strom (DecisionTreeClassifier)."""
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
) -> dict:
    """Spustí a vyhodnotí náhodný les (RandomForestClassifier)."""
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
    )


def run_svm(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: str = "scale",
    random_state: int = 42,
    save_plots: bool = True,
) -> dict:
    """
    Spustí a vyhodnotí SVM s nelineárním RBF jádrem (SVC).
    Očekává data již standardizovaná z data_prep.py (scale=True).
    """
    model = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
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
    )


def compare_results(results: list[dict], save_csv_path: str | None = "grafy/vysledky_modelu.csv") -> pd.DataFrame:
    """Sestaví souhrnnou srovnávací tabulku všech modelů a volitelně ji uloží do CSV."""
    rows = []
    for res in results:
        rows.append({
            "Model": res["model_name"],
            "Accuracy (%)": round(res["accuracy"] * 100, 2),
            "Precision (%)": round(res["precision"] * 100, 2),
            "Recall (%)": round(res["recall"] * 100, 2),
            "F1-score (%)": round(res["f1_score"] * 100, 2),
            "TP": res["TP"],
            "TN": res["TN"],
            "FP": res["FP"],
            "FN": res["FN"],
            "Fit čas (s)": round(res["fit_time_s"], 4),
            "Predict čas (s)": round(res["predict_time_s"], 4),
        })

    summary_df = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print("SOUHRNNÉ POROVNÁNÍ MODELŮ")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)

    if save_csv_path:
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
        summary_df.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] Souhrnná tabulka uložena do: {save_csv_path}")

    return summary_df
