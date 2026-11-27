import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def uloz_matici_zamen(cm, nazev_modelu, nazev_souboru):
    os.makedirs("grafy", exist_ok=True)
    tn, fp, fn, tp = cm.ravel()
    popisky = [[f"TN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"TP\n{tp}"]]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=popisky,
        fmt="",
        cmap="Blues",
        xticklabels=["Nespokojený (0)", "Spokojený (1)"],
        yticklabels=["Nespokojený (0)", "Spokojený (1)"],
    )
    plt.title(f"Matice záměn – {nazev_modelu}")
    plt.xlabel("Predikováno")
    plt.ylabel("Skutečnost")
    plt.tight_layout()
    plt.savefig(f"grafy/{nazev_souboru}", dpi=300)
    plt.close()


# ============================================================
# 1. DECISION TREE
# ============================================================
def decision_tree(X_train, X_test, y_train, y_test):
    print("--- Decision Tree ---")
    model = DecisionTreeClassifier(max_depth=12, random_state=42)

    start = time.time()
    model.fit(X_train, y_train)
    cas_fit = time.time() - start

    start = time.time()
    y_pred = model.predict(X_test)
    cas_predict = time.time() - start

    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred) * 100
    rec = recall_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"Accuracy: {acc:.2f} % | Precision: {prec:.2f} % | Recall: {rec:.2f} % | F1: {f1:.2f} %")
    print(f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
    print(f"Čas fit: {cas_fit:.2f} s | Čas predict: {cas_predict:.4f} s\n")

    # Uložení matice záměn
    uloz_matici_zamen(cm, "Decision Tree", "cm_decision_tree.png")

    # Vykreslení prvních 3 pater rozhodovacího stromu (čitelný náhled na poster)
    plt.figure(figsize=(22, 10))
    plot_tree(
        model,
        max_depth=3,
        feature_names=list(X_train.columns),
        class_names=["Nespokojený", "Spokojený"],
        filled=True,
        rounded=True,
        fontsize=9,
    )
    plt.title("Rozhodovací strom (horní 3 úrovně)", fontsize=16)
    plt.tight_layout()
    plt.savefig("grafy/strom_decision_tree.png", dpi=300)
    plt.close()

    return {
        "Model": "Decision Tree",
        "Accuracy (%)": acc,
        "Precision (%)": prec,
        "Recall (%)": rec,
        "F1 (%)": f1,
        "Fit (s)": cas_fit,
        "Predict (s)": cas_predict,
    }


# ============================================================
# 2. RANDOM FOREST
# ============================================================
def random_forest(X_train, X_test, y_train, y_test):
    print("--- Random Forest ---")
    model = RandomForestClassifier(max_depth=20, random_state=42, n_jobs=-1)

    start = time.time()
    model.fit(X_train, y_train)
    cas_fit = time.time() - start

    start = time.time()
    y_pred = model.predict(X_test)
    cas_predict = time.time() - start

    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred) * 100
    rec = recall_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"Accuracy: {acc:.2f} % | Precision: {prec:.2f} % | Recall: {rec:.2f} % | F1: {f1:.2f} %")
    print(f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
    print(f"Čas fit: {cas_fit:.2f} s | Čas predict: {cas_predict:.4f} s")

    # Uložení matice záměn
    uloz_matici_zamen(cm, "Random Forest", "cm_random_forest.png")

    # Top 10 nejdůležitějších příznaků
    dulezitost = pd.Series(model.feature_importances_, index=X_train.columns)
    print("Top 5 příznaků:")
    print(dulezitost.nlargest(5).round(4), "\n")

    top10 = dulezitost.nlargest(10).sort_values()
    plt.figure(figsize=(9, 6))
    plt.barh(top10.index, top10.values, color="steelblue")
    plt.title("Top 10 nejdůležitějších příznaků (Random Forest)")
    plt.xlabel("Důležitost příznaku")
    plt.tight_layout()
    plt.savefig("grafy/dulezitost_random_forest.png", dpi=300)
    plt.close()

    return {
        "Model": "Random Forest",
        "Accuracy (%)": acc,
        "Precision (%)": prec,
        "Recall (%)": rec,
        "F1 (%)": f1,
        "Fit (s)": cas_fit,
        "Predict (s)": cas_predict,
    }


# ============================================================
# 3. SVM (RBF)
# ============================================================
def svm(X_train, X_test, y_train, y_test):
    print("--- SVM (RBF) ---")
    # StandardScaler je jen v pipeline pro SVM, aby se škálovalo výhradně na X_train
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", cache_size=1000, random_state=42),
    )

    start = time.time()
    model.fit(X_train, y_train)
    cas_fit = time.time() - start

    start = time.time()
    y_pred = model.predict(X_test)
    cas_predict = time.time() - start

    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred) * 100
    rec = recall_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"Accuracy: {acc:.2f} % | Precision: {prec:.2f} % | Recall: {rec:.2f} % | F1: {f1:.2f} %")
    print(f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
    print(f"Čas fit: {cas_fit:.2f} s | Čas predict: {cas_predict:.4f} s\n")

    # Uložení matice záměn
    uloz_matici_zamen(cm, "SVM (RBF)", "cm_svm.png")

    return {
        "Model": "SVM (RBF)",
        "Accuracy (%)": acc,
        "Precision (%)": prec,
        "Recall (%)": rec,
        "F1 (%)": f1,
        "Fit (s)": cas_fit,
        "Predict (s)": cas_predict,
    }