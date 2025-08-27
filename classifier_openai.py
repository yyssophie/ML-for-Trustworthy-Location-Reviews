import os, re, json, argparse
import pandas as pd
from openai import OpenAI
import os


prompt = (
    "You are a super great classifier.\n"
    "You should read a line of review for a location\n"
    "which contains 5 attributes: Business (the name of that place),\n"
    "Author (the name of the review writer), Rating,\n"
    "Category and Comment (most important part to analyze)\n"
    "You should then decide which of the following 4 types this review should fall into:\n"
    "advertisement, irrelevant, rants_from_non_visitors and reasonable_comment.\n\n"
    "Some of the examples are as follows:\n"
    "“Best pizza! Visit www.pizzapromo.com for discounts!”. This review is an advertisement, as it contains promotional content and links\n"
    "“I love my new phone, but this place is too noisy.”. This review is irrelevant, as reviews must be about the location, not unrelated topics.\n"
    "“Never been here, but I heard it’s terrible.”. This review is rants, as it doesn't come from actual visitors\n"
    "Output only the label, with no extra text."
)

client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
df = pd.read_csv("reviews_for_llm.csv", header=None, names=["text"])
labels = []
for txt in df['text']:
    content = str(txt or "").strip()
    if not content:
        labels.append("irrelevant")
        continue

    completion = client.chat.completions.create(
        model="qwen-plus",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=[
            {'role': 'user', 'content': prompt},
            {'role': 'user', 'content': txt}
        ]
    )
    label = (completion.choices[0].message.content or "").strip()
    labels.append(label)
    print(txt, label)

df['label'] = labels
df.to_csv("reviews_with_labels.csv", index=False, header=False)
df.to_json("reviews_with_labels.json", orient="records", force_ascii=False, indent=2)





