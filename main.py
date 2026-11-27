import sys
from pathlib import Path

def find_project_dir(target_file: str = "data_prep.py") -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent

    cwd = Path.cwd()
    candidates = [
        cwd,
        *[p.parent for p in cwd.glob(f"*/{target_file}")],
        *[p.parent for p in cwd.glob(f"*/*/{target_file}")],
        *list(cwd.parents)[:2],
    ]
    for folder in candidates:
        if (folder / target_file).exists():
            return folder

    obsah = [p.name for p in cwd.iterdir()]
    raise FileNotFoundError(
        f"Soubor '{target_file}' nebyl nalezen.\n"
        f"Aktuální složka (cwd): {cwd}\n"
        f"Soubory v této složce: {obsah}"
    )

BASE_DIR = find_project_dir("data_prep.py")
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

print(f"[INFO] Projektová složka: {BASE_DIR}")

from data_prep import prepare_data  # noqa: E402
from MethodsML import (  # noqa: E402
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
TRAIN_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"

# Pro rychlé ověření funkčnosti (hlavně kvůli SVM) nastav např. 15000.
# Pro finální běh a výsledky na poster ponech None (celý dataset).
SAMPLE_SIZE = None

SAVE_PLOTS = True
PLOTS_DIR = BASE_DIR / "grafy"
RESULTS_CSV_PATH = BASE_DIR / "vysledky" / "souhrn_modelu.csv"


def save_results_to_csv(summary_df: pd.DataFrame, output_path: Union[Path, str]) -> None:
    """Uloží souhrnnou tabulku výsledků do CSV (utf-8-sig kvůli správné diakritice v Excelu)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        summary_df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[INFO] Souhrnné výsledky uloženy do: {path}")
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_{int(time.time())}{path.suffix}")
        summary_df.to_csv(fallback, index=False, encoding="utf-8-sig")
        print(
            f"[VAROVÁNÍ] Soubor '{path}' je pravděpodobně otevřený v Excelu. "
            f"Výsledky uloženy do záložního souboru: {fallback}"
        )


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
            train_path=str(TRAIN_PATH),
            test_path=str(TEST_PATH),
            scale=True,
            sample_size=SAMPLE_SIZE,
            random_state=RANDOM_SEED,
            save_corr_plot=SAVE_PLOTS,
        )
    except FileNotFoundError as exc:
        print(f"[CHYBA] Vstupní dataset nebyl nalezen: {exc}")
        print(f"Zkontroluj, že soubory '{TRAIN_PATH}' a '{TEST_PATH}' existují.")
        return

    print(f"[INFO] Data připravena: X_train={X_train.shape}, X_test={X_test.shape}\n")

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
        output_dir=str(PLOTS_DIR),
    )

    # 3. Výpis souhrnného porovnání a export do CSV pro poster
    print("\n" + "=" * 60)
    print("3. SOUHRNNÉ POROVNÁNÍ MODELŮ")
    print("=" * 60)
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(summary_df.to_string(index=False))

    save_results_to_csv(summary_df, RESULTS_CSV_PATH)

    total_elapsed = time.perf_counter() - total_start
    print(f"\n[HOTOVO] Celková doba běhu experimentu: {total_elapsed:.2f} s")


if __name__ == "__main__":
    main()