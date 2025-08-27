import os, re, json, argparse
import pandas as pd
from openai import OpenAI
import google.generativeai as genai

Prompt = (
    "You are a super great classifier.\n"
    "You should read a line of review on google map for a location\n"
    "which contains 5 attributes: Business (the name of that place),\n"
    "Author (the name of the review writer), Rating (the rating of the review),\n"
    "Category and Comment (most important part to analyze)\n"
    "You should then decide which of the following 4 types this review should fall into:\n"
    "advertisement, irrelevant, rants and reasonable_comment.\n\n"
    "Some of the examples are as follows to help you understand how to classify them:\n"
    "“Best pizza! Visit www.pizzapromo.com for discounts!”. This review is an advertisement, as it contains promotional content and links\n"
    "“I love my new phone, but this place is too noisy.”. This review is irrelevant, as reviews must be about the location, not unrelated topics.\n"
    "“Never been here, but I heard it’s terrible.”. This review is rants, as this negative review doesn't come from actual visitors\n"
    "Output only the label(i.e. classification) for the you read, with no extra text."
)

genai.configure(api_key="")
model = genai.GenerativeModel(
    "gemini-1.5-flash"
)
gen_cfg = genai.types.GenerationConfig(temperature=0.0)

df = pd.read_csv("reviews_for_llm.csv", header=None, names=["text"]).sample(1)
labels = []
for txt in df['text']:
    content = str(txt or "").strip()
    if not content:
        labels.append("irrelevant")
        continue
    prompt = (
        "You are a super great classifier.\n"
        "You should read a line of review on google map for a location\n"
        "which contains 5 attributes: Business (the name of that place),\n"
        "Author (the name of the review writer), Rating (the rating of the review),\n"
        "Category and Comment (most important part to analyze)\n"
        "You should then decide which of the following 4 types this review should fall into:\n"
        "advertisement, irrelevant, rants and reasonable_comment.\n\n"
        "Some of the examples are as follows to help you understand how to classify them:\n"
        "“Best pizza! Visit www.pizzapromo.com for discounts!”. This review is an advertisement, as it contains promotional content and links\n"
        "“I love my new phone, but this place is too noisy.”. This review is irrelevant, as reviews must be about the location, not unrelated topics.\n"
        "“Never been here, but I heard it’s terrible.”. This review is rants, as this negative review doesn't come from actual visitors\n"
        "Output only the label(i.e. classification) for the you read, with no extra text."
        f"Review: {txt}"
    )
    completion = model.generate_content(prompt)
    label = (completion.text or "").strip()
    labels.append(label)

print(labels)

