import os
import json
import pandas as pd
from openai import OpenAI
import random

'''
def sample_reviews(path, n=100, seed=42):
    reviews = []
    with open(path, 'rt', encoding='utf-8') as f:  # 'rt' 表示文本模式
        for line in f:
            try:
                review = json.loads(line)
                if review.get("text") and review["text"].strip():  # 过滤掉空评论
                    reviews.append(review)
            except json.JSONDecodeError:
                continue  # 避免坏行报错

    print(f"总评论数: {len(reviews)}")

    # 固定随机种子，保证结果可复现
    random.seed(seed)
    sampled = random.sample(reviews, min(n, len(reviews)))
    return sampled

# 读取项目目录下的 reviews.json.gz
sampled_reviews = sample_reviews("review-Alabama_10.json", n=100)

# 转换成 DataFrame 并保存
df = pd.DataFrame(sampled_reviews)
df.to_csv("sampled_100_reviews.csv", index=False)
print("已保存 sampled_100_reviews.csv")


reviews_df = pd.read_csv("sampled_100_reviews.csv")
places_data = []
with open("meta-Alabama.json", "rt", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        try:
            places_data.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"⚠️ 跳过坏行 {i}")
            continue

places_df = pd.DataFrame(places_data)

# 确认 gmap_id 在两个表里都是字符串
reviews_df['gmap_id'] = reviews_df['gmap_id'].astype(str)
places_df['gmap_id'] = places_df['gmap_id'].astype(str)

# inner join（只保留两个表都有 gmap_id 的）
merged_df = reviews_df.merge(places_df, on="gmap_id", how="inner")

print(f"合并后数据量: {len(merged_df)}")
print(merged_df.head(3))

# 保存结果
merged_df.to_csv("reviews_with_places.csv", index=False)

'''
df = pd.read_csv("reviews_with_places.csv")

# 检查 category 列的缺失情况
print("总行数:", len(df))
print("description 缺失数量:", df['description'].isnull().sum())
print("description 非缺失数量:", df['description'].notnull().sum())

# 如果想看是否每行都有 category
if df['description'].isnull().sum() == 0:
    print("✅ 所有行都有 category")
else:
    print("⚠️ 有缺失 category 的行")