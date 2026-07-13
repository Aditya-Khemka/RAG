from pathlib import Path
import sys

import polars as pl

# Add the project root (RAG/) to Python's path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# CSV path check
csv_path = project_root / "data" / "Articles.csv"
if not csv_path.exists():
    csv_path = project_root / "data" / "articles.csv"
if not csv_path.exists():
    raise FileNotFoundError(
        f"CSV file not found at expected locations: {project_root / 'data' / 'articles.csv'} or {project_root / 'data' / 'Articles.csv'}"
    )

# Load with Polars using cp1252 encoding to handle Windows-style smart quotes
try:
    articles = pl.read_csv(csv_path, encoding="cp1252").with_row_index("id" , offset=1)
    print(f"Loaded CSV with polars from {csv_path}")
    print(articles.head())
except Exception as error:
    raise RuntimeError(f"Polars failed to read CSV at {csv_path}: {error}") from error

articles = articles[:25]
print(f"Loaded {len(articles)} articles from CSV.")

# for article in articles.iter_rows(named=True):
#     # Print the Column 'Date' and 'heading' for each article
#     # print(f"ID : {article["id"]} , content : {article["Article"]}")
#     # print(f"Date: {article['Date']}, Heading: {article['Heading']}")