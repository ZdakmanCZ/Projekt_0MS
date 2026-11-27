import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def plot_correlation_matrix(df: pd.DataFrame, output_path: str = "grafy/korelacni_matice.png") -> None:
    """Vykreslí a uloží korelační matici číselných sloupců pro poster."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    plt.figure(figsize=(14, 11))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        annot=False,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Korelační matice vstupních znaků (Airline Passenger Satisfaction)", fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Korelační matice uložena do: {output_path}")


def print_eda_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Vypíše základní průzkum a statistiky datasetu (bod 3 zadání)."""
    print("=" * 60)
    print("1. ZÁKLADNÍ PŘEHLED DATASETU")
    print("=" * 60)
    print(f"Rozměry trénovací sady: {train_df.shape[0]} řádků, {train_df.shape[1]} sloupců")
    print(f"Rozměry testovací sady: {test_df.shape[0]} řádků, {test_df.shape[1]} sloupců")

    print("\nChybějící hodnoty (NaN) v trénovací sadě:")
    missing_train = train_df.isnull().sum()
    print(missing_train[missing_train > 0] if (missing_train > 0).any() else "Žádné")

    print("\nChybějící hodnoty (NaN) v testovací sadě:")
    missing_test = test_df.isnull().sum()
    print(missing_test[missing_test > 0] if (missing_test > 0).any() else "Žádné")

    print("\nRozložení cílové proměnné (satisfaction) v trénovací sadě:")
    target_counts = train_df["satisfaction"].value_counts(normalize=True) * 100
    for label, pct in target_counts.items():
        print(f"  - {label}: {pct:.2f} %")

    print("\nZákladní popisné statistiky vybraných číselných sloupců:")
    cols_to_describe = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
    print(train_df[cols_to_describe].describe().round(2))
    print("=" * 60)


def prepare_data(
    train_path: str = "data/train.csv",
    test_path: str = "data/test.csv",
    scale: bool = True,
    sample_size: int | None = None,
    random_state: int = 42,
    save_corr_plot: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Načte, vyčistí a předzpracuje dataset Airline Passenger Satisfaction.

    Parametry:
        train_path: Cesta k trénovacímu CSV souboru.
        test_path: Cesta k testovacímu CSV souboru.
        scale: Pokud True, provede StandardScaler (fit pouze na trénovacích datech).
        sample_size: Volitelný počet řádků z train sady pro rychlé testování modelů (např. 10000).
        random_state: Seed pro reprodukovatelnost.
        save_corr_plot: Pokud True, uloží heatmapu korelací do složky grafy/.

    Vrací:
        X_train, X_test, y_train, y_test
    """
    # 1. Načtení dat
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Volitelné zmenšení vzorku pro rychlé ladění pomalejších modelů (např. SVC / k-NN)
    if sample_size is not None and sample_size < len(train_df):
        train_df = train_df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        print(f"[INFO] Použit zmenšený trénovací vzorek: {sample_size} řádků.")

    # 2. Průzkumná analýza (EDA)
    print_eda_summary(train_df, test_df)

    # 3. Odstranění zbytečných ID sloupců
    id_cols = ["Unnamed: 0", "id"]
    train_df = train_df.drop(columns=id_cols, errors="ignore")
    test_df = test_df.drop(columns=id_cols, errors="ignore")

    # 4. Převod cílové proměnné na binární 0 / 1
    target_mapping = {
        "neutral or dissatisfied": 0,
        "satisfied": 1,
    }
    y_train = train_df["satisfaction"].map(target_mapping)
    y_test = test_df["satisfaction"].map(target_mapping)

    if y_train.isnull().any() or y_test.isnull().any():
        raise ValueError("Ve sloupci 'satisfaction' se objevily neočekávané hodnoty.")

    X_train = train_df.drop(columns=["satisfaction"])
    X_test = test_df.drop(columns=["satisfaction"])

    # 5. Doplnění chybějících hodnot (medián spočítaný pouze z trénovací sady -> bez úniku dat)
    if "Arrival Delay in Minutes" in X_train.columns:
        median_arrival_delay = X_train["Arrival Delay in Minutes"].median()
        X_train["Arrival Delay in Minutes"] = X_train["Arrival Delay in Minutes"].fillna(median_arrival_delay)
        X_test["Arrival Delay in Minutes"] = X_test["Arrival Delay in Minutes"].fillna(median_arrival_delay)
        print(f"[INFO] Chybějící hodnoty v 'Arrival Delay in Minutes' doplněny mediánem: {median_arrival_delay}")

    # 6. Korelační analýza a odstranění silně korelovaného sloupce
    if save_corr_plot:
        plot_correlation_matrix(X_train)

    if "Arrival Delay in Minutes" in X_train.columns and "Departure Delay in Minutes" in X_train.columns:
        corr_val = X_train["Departure Delay in Minutes"].corr(X_train["Arrival Delay in Minutes"])
        print(f"[INFO] Korelace Departure vs. Arrival Delay: {corr_val:.4f} -> odstraňuji 'Arrival Delay in Minutes'.")
        X_train = X_train.drop(columns=["Arrival Delay in Minutes"])
        X_test = X_test.drop(columns=["Arrival Delay in Minutes"])

    # 7. Převod kategorických sloupců na číselné vektory (One-Hot Encoding)
    categorical_cols = ["Gender", "Customer Type", "Type of Travel", "Class"]
    X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True, dtype=int)
    X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True, dtype=int)

    # Sjednocení sloupců mezi train a test sadou (pro jistotu, kdyby v testu některá kategorie chyběla)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # 8. Standardizace dat (StandardScaler)
    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )
        print("[INFO] Data byla standardizována pomocí StandardScaler (fit pouze na train).")

    print(f"[INFO] Finální tvar dat -> X_train: {X_train.shape}, X_test: {X_test.shape}\n")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Rychlý test modulu při přímém spuštění
    X_train, X_test, y_train, y_test = prepare_data(
        train_path="data/train.csv",
        test_path="data/test.csv",
        scale=True,
    )