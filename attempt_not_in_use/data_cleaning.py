import pandas as pd

df = pd.read_csv("reviews.csv")

# make sure rating is numeric
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

df["llm_input"] = (
    "Business: " + df["business_name"].astype(str).fillna("") +
    " | Author: " + df["author_name"].astype(str).fillna("") +
    " | Rating: " + df["rating"].astype(str) +
    " | Category: " + df["rating_category"].astype(str).fillna("") +
    " | Comment: " + df["text"].astype(str).fillna("")
)

df[["llm_input"]].to_csv("bad_reviews.csv", index=False, header=False)
