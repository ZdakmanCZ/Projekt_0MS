import pandas as pd
import matplotlib.pyplot as plt
from data_prep import priprav_data
from MethodsML import decision_tree, random_forest, svm


# 1. Příprava dat
X_train, X_test, y_train, y_test = priprav_data("data/train.csv", "data/test.csv")

# 2. Spuštění a vyhodnocení všech 3 modelů
vysledky = [
    decision_tree(X_train, X_test, y_train, y_test),
    random_forest(X_train, X_test, y_train, y_test),
    svm(X_train, X_test, y_train, y_test),
]

# 3. Souhrnná tabulka v terminálu
tabulka = pd.DataFrame(vysledky)
print("=== SOUHRNNÉ SROVNÁNÍ MODELŮ ===")
print(tabulka.round(2).to_string(index=False))

# 4. Souhrnný graf pro poster (přesnost a čas trénování vedle sebe)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

tabulka.set_index("Model")[["Accuracy (%)", "F1 (%)"]].plot(
    kind="bar", ax=ax1, color=["steelblue", "darkorange"], rot=0
)
ax1.set_title("Porovnání přesnosti modelů")
ax1.set_ylabel("Úspěšnost (%)")
ax1.set_ylim(85, 100)
for kontejner in ax1.containers:
    ax1.bar_label(kontejner, fmt="%.2f", fontsize=8)

tabulka.set_index("Model")["Fit (s)"].plot(
    kind="bar", ax=ax2, color="seagreen", rot=0
)
ax2.set_title("Čas trénování modelů (fit)")
ax2.set_ylabel("Čas (sekundy)")
for kontejner in ax2.containers:
    ax2.bar_label(kontejner, fmt="%.1f s", fontsize=9)

plt.tight_layout()
plt.savefig("grafy/srovnani_modelu.png", dpi=300)
plt.close()
print("\nHotovo! Všechny grafy byly uloženy do složky 'grafy/'.")