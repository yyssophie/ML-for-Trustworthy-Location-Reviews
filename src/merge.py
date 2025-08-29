import pandas as pd
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import sys

lab = pd.read_csv("data/out/label_reviews_Kentucky.csv")

places_data = []
with open("data/in/meta-Kentucky.json", "rt", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        try:
            places_data.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"⚠️ 跳过坏行 {i}")
            continue

places_df = pd.DataFrame(places_data)

merged_df = lab.merge(
    places_df
    , left_on="business_name", right_on="name", how="left")

merged_df.to_csv("data/out/merged_label_reviews_Kentucky.csv")