"""Load the CFPB complaints (with narratives), merge renamed labels, save a clean parquet."""
import glob
import pandas as pd

# CFPB renamed/merged categories over time. Map every name to one canonical label.
CANON = {
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
    "Credit reporting or other personal consumer reports": "Credit reporting",
    "Credit reporting": "Credit reporting",
    "Credit card or prepaid card": "Credit/prepaid card",
    "Credit card": "Credit/prepaid card",
    "Prepaid card": "Credit/prepaid card",
    "Checking or savings account": "Bank account",
    "Bank account or service": "Bank account",
    "Money transfer, virtual currency, or money service": "Money transfer",
    "Money transfers": "Money transfer",
    "Virtual currency": "Money transfer",
    "Payday loan, title loan, or personal loan": "Personal/payday loan",
    "Payday loan, title loan, personal loan, or advance loan": "Personal/payday loan",
    "Payday loan": "Personal/payday loan",
    "Consumer Loan": "Personal/payday loan",
    "Debt collection": "Debt collection",
    "Mortgage": "Mortgage",
    "Student loan": "Student loan",
    "Vehicle loan or lease": "Vehicle loan or lease",
}  # "Other financial service" and "Debt or credit management" are tiny and dropped.

if __name__ == "__main__":
    cols = ["Date received", "Product", "Consumer complaint narrative", "Company"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/hf/has-text-*.parquet"))],
                   ignore_index=True)
    df.columns = ["date", "product", "text", "company"]
    df["label"] = df["product"].map(CANON)
    df = df.dropna(subset=["label"]).copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df = df.drop_duplicates(subset="text")
    df[["year", "label", "company", "text"]].to_parquet("data/clean.parquet")
    print(len(df)); print(df["label"].value_counts().to_string())
