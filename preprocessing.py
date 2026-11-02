"""
Preprocesare - Machine Predictive Maintenance Classification
Scenariu identic cu procesul din Altair AI Studio:
Nominal to Numerical -> Filter Examples -> Generate Attributes ->
Select Attributes -> Rename -> Numerical to Binominal -> Set Role -> Normalize

Rezultat: predictive_maintenance_clean.csv
"""
import pandas as pd
import matplotlib.pyplot as plt

FISIER_INTRARE = "predictive_maintenance.csv"
FISIER_IESIRE = "clean_predictive_maintenance.csv"


def load_data(path: str) -> pd.DataFrame:
    """Încarcă setul de date din fișierul CSV (echivalent: Retrieve)."""
    return pd.read_csv(path)


def check_quality(df: pd.DataFrame) -> None:
    """Afișează valorile lipsă, duplicatele și distribuția clasei țintă."""
    print("Valori lipsă:\n", df.isna().sum())
    print("Duplicate:", df.duplicated().sum())
    print("Distribuție Target:\n", df["Target"].value_counts(normalize=True))


def nominal_to_numerical(df: pd.DataFrame) -> pd.DataFrame:
    """Dummy coding pe atributul Type: L, M, H devin coloane binare 0/1
    (echivalent: Nominal to Numerical, attribute filter type = single,
    coding type = dummy coding)."""
    return pd.get_dummies(df, columns=["Type"], dtype=int)


def filter_examples(df: pd.DataFrame) -> pd.DataFrame:
    """Elimină rândurile contradictorii: Target = 1 și Failure Type = No Failure
    (echivalent: Filter Examples cu invert filter)."""
    contradictoriu = (df["Target"] == 1) & (df["Failure Type"] == "No Failure")
    return df[~contradictoriu].copy()


def generate_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Creează Target_new: 1 pentru Random Failures, altfel valoarea inițială
    (echivalent: Generate Attributes cu if([Failure Type]=="Random Failures", 1, [Target]))."""
    df["Target_new"] = df["Target"].where(df["Failure Type"] != "Random Failures", 1)
    return df


def select_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Elimină identificatorii, Failure Type (data leakage) și vechiul Target
    (echivalent: Select Attributes cu invert selection)."""
    return df.drop(columns=["UDI", "Product ID", "Failure Type", "Target"])


def rename_target(df: pd.DataFrame) -> pd.DataFrame:
    """Redenumește Target_new în Target (echivalent: Rename)."""
    return df.rename(columns={"Target_new": "Target"})


def numerical_to_binominal(df: pd.DataFrame) -> pd.DataFrame:
    """Convertește Target în atribut binominal: 0 -> false, 1 -> true
    (echivalent: Numerical to Binominal). Rolul de label este dat de separarea
    ulterioară a lui y (Set Role)."""
    df["Target"] = df["Target"].map({0: "false", 1: "true"})
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizare Z-score doar pe atributele senzorilor.
    Sunt excluse label-ul (Target) și coloanele dummy Type_* (echivalent:
    Normalize cu Z-transformation, pe un subset de atribute)."""
    cols = [c for c in df.columns
            if c != "Target" and not c.startswith("Type_")]
    df[cols] = (df[cols] - df[cols].mean()) / df[cols].std()
    return df


if __name__ == "__main__":
    df = load_data(FISIER_INTRARE)
    check_quality(df)
    print("Dimensiune inițială:", df.shape)

    df = nominal_to_numerical(df)
    df = filter_examples(df)
    df = generate_attributes(df)
    df = select_attributes(df)
    df = rename_target(df)
    df = numerical_to_binominal(df)

    # Set Role: separăm atributele (X) de label (y)
    X = df.drop(columns=["Target"])
    y = df["Target"]
    print("Distribuție label:\n", y.value_counts())

    df = normalize(df)

    # Verificare vizuală a distribuțiilor după normalizare
    df.drop(columns=["Target"]).hist(figsize=(10, 8))
    plt.tight_layout()
    plt.savefig("distributii.png")

    df.to_csv(FISIER_IESIRE, index=False)
    print("Set final:", df.shape)