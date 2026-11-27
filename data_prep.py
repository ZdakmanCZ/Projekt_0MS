import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def priprav_data(train_path="data/train.csv", test_path="data/test.csv"):
    os.makedirs("grafy", exist_ok=True)

    # 1. Načtení dat a smazání technických sloupců
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df = train_df.drop(columns=["Unnamed: 0", "id"], errors="ignore")
    test_df = test_df.drop(columns=["Unnamed: 0", "id"], errors="ignore")

    # 2. Základní přehled datasetu (EDA)
    print("=== PŘEHLED DATASETU ===")
    print(f"Train: {train_df.shape[0]} řádků, {train_df.shape[1]} sloupců")
    print(f"Test:  {test_df.shape[0]} řádků, {test_df.shape[1]} sloupců")
    print(f"Duplicity (train / test): {train_df.duplicated().sum()} / {test_df.duplicated().sum()}")

    print("\nChybějící hodnoty v train:")
    chybejici = train_df.isnull().sum()
    print(chybejici[chybejici > 0])

    print("\nRozložení cílové proměnné (satisfaction):")
    print((train_df["satisfaction"].value_counts(normalize=True) * 100).round(2))

    # 3. Doplnění chybějících hodnot mediánem z trénovací sady (bez úniku dat)
    median_zpozdeni = train_df["Arrival Delay in Minutes"].median()
    train_df["Arrival Delay in Minutes"] = train_df["Arrival Delay in Minutes"].fillna(median_zpozdeni)
    test_df["Arrival Delay in Minutes"] = test_df["Arrival Delay in Minutes"].fillna(median_zpozdeni)

    # 4. Korelační matice číselných sloupců (před zakódováním kategorií)
    ciselne = train_df.select_dtypes(include="number")
    korelace = ciselne.corr()

    r = korelace.loc["Departure Delay in Minutes", "Arrival Delay in Minutes"]
    print(f"\nKorelace Departure vs. Arrival Delay: {r:.3f} -> vyřazuji Arrival Delay")

    plt.figure(figsize=(14, 11))
    sns.heatmap(korelace, cmap="coolwarm", center=0, annot=True, fmt=".2f", annot_kws={"size": 7})
    plt.title("Korelační matice číselných příznaků")
    plt.tight_layout()
    plt.savefig("grafy/korelacni_matice.png", dpi=300)
    plt.close()

    # Vyřazení redundantního sloupce kvůli silné korelaci
    train_df = train_df.drop(columns=["Arrival Delay in Minutes"])
    test_df = test_df.drop(columns=["Arrival Delay in Minutes"])

    # Pro rychlý test kódu (kvůli pomalejšímu SVM) odkomentuj řádek níže:
    # train_df = train_df.sample(15000, random_state=42)

    # 5. Oddělení vstupů (X) a cílové proměnné (y: 1 = satisfied, 0 = neutral/dissatisfied)
    y_train = (train_df["satisfaction"] == "satisfied").astype(int)
    y_test = (test_df["satisfaction"] == "satisfied").astype(int)

    X_train = train_df.drop(columns=["satisfaction"])
    X_test = test_df.drop(columns=["satisfaction"])

    # 6. Zakódování textových kategorií (One-Hot Encoding) a srovnání sloupců
    X_train = pd.get_dummies(X_train, drop_first=True, dtype=int)
    X_test = pd.get_dummies(X_test, drop_first=True, dtype=int)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    print(f"Data připravena: X_train {X_train.shape}, X_test {X_test.shape}\n")
    return X_train, X_test, y_train, y_test