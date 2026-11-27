import os
import time
import pandas as pd

from data_prep import prepare_data
from MethodsML import (
    run_all_models,
    # Pro samostatné volání a ladění parametrů jednotlivých modelů lze odkomentovat:
    # run_decision_tree,
    # run_random_forest,
    # run_svm,
)

# ==============================================================================
# KONFIGURACE EXPERIMENTU
# ==============================================================================
RANDOM_SEED = 42
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

# Pro rychlé ověření funkčnosti (hlavně kvůli SVM) nastav např. 15000.
# Pro finální běh a výsledky na poster ponech None (celý dataset).
SAMPLE_SIZE = None

SAVE_PLOTS = True
RESULTS_CSV_PATH = "vysledky/souhrn_modelu.csv"


def save_results_to_csv(summary_df: pd.DataFrame, output_path: str) -> None:
    """Uloží souhrnnou tabulku výsledků do CSV (utf-8-sig kvůli správné diakritice v Excelu)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Souhrnné výsledky uloženy do: {output_path}")


def main() -> None:
    total_start = time.perf_counter()

    print("=" * 60)
    print("PROJEKT 0MS: AIRLINE PASSENGER SATISFACTION")
    print("=" * 60)
    print(
        f"Konfigurace: RANDOM_SEED={RANDOM_SEED} | "
        f"SAMPLE_SIZE={SAMPLE_SIZE} | SAVE_PLOTS={SAVE_PLOTS}\n"
    )

    # 1. Načtení, vyčištění, EDA a škálování dat (data_prep.py)
    try:
        X_train, X_test, y_train, y_test = prepare_data(
            train_path=TRAIN_PATH,
            test_path=TEST_PATH,
            scale=True,
            sample_size=SAMPLE_SIZE,
            random_state=RANDOM_SEED,
            save_corr_plot=SAVE_PLOTS,
        )
    except FileNotFoundError as exc:
        print(f"[CHYBA] Vstupní dataset nebyl nalezen: {exc}")
        print("Zkontroluj, že soubory 'data/train.csv' a 'data/test.csv' existují.")
        return

    print(f"\n[INFO] Data připravena: X_train={X_train.shape}, X_test={X_test.shape}\n")

    # 2. Trénování a vyhodnocení modelů: Decision Tree, Random Forest, SVM (MethodsML.py)
    print("=" * 60)
    print("2. TRÉNOVÁNÍ A VYHODNOCENÍ MODELŮ")
    print("=" * 60)

    summary_df = run_all_models(
        X_train,
        X_test,
        y_train,
        y_test,
        random_state=RANDOM_SEED,
        save_plots=SAVE_PLOTS,
    )

    # 3. Výpis souhrnného porovnání a export do CSV pro poster
    print("\n" + "=" * 60)
    print("3. SOUHRNNÉ POROVNÁNÍ MODELŮ")
    print("=" * 60)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print(summary_df.to_string(index=False))

    save_results_to_csv(summary_df, RESULTS_CSV_PATH)

    total_elapsed = time.perf_counter() - total_start
    print(f"\n[HOTOVO] Celková doba běhu experimentu: {total_elapsed:.2f} s")


if __name__ == "__main__":
    main()
