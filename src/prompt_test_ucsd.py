import pandas as pd
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


ENGLSIH_SYS_PRMOPT="""
You are a top-tier content moderation expert specializing in the evaluation of Google Maps location reviews. 
Your task is to parse a JSON object containing review data and accurately classify it according to the following policies and rules.

# Moderation Policies & Label Definitions:

1. **"Valid"**  
   A normal review that is relevant to the location and shares a genuine experience.

2. **"Advertisement"**  
   The primary purpose of the review is to promote another product, service, or website.  
   - Detect **URL / links** (e.g., "http://", "www.").  
   - Detect **promotion keywords** (e.g., "discount", "promo", "sale").  

3. **"Irrelevant"**  
   The review content is completely unrelated to the location, service, or experience being reviewed.  
   - **Relevancy Rules**:  
     - If short text (≤ N words) → need to carefully check relevance. 
     - If the subject or adjective in the short text does not relate to the business description, category, or name, classify as Irrelevant.
     - Check relevancy order:  
       1. Compare review text with **description** (if available).  
       2. Compare review text with **category** (if available).  
       3. Compare review text with **business name** (always available).  
     - Note: All reviews have category and name, but description may be missing.  
     - Additional signal: Extreme star rating + vague/muffled comment → may indicate low relevancy.  

4. **"Rant_Without_Visit"**  
   The review is filled with anger or complaints, but the content explicitly states or strongly implies the user has never actually visited the location.  
   - Check rant signals:  
     - Contains phrases like "never been", "haven’t visited", "I heard", "my friend told me".  
     - Review sentiment is strongly negative, but no direct experience is described.  
     - Often accompanied by extreme star rating (e.g., 1⭐) with no supporting details.  

---

# Examples:
The following are correctly classified examples. Please learn from them to guide your judgment.

---
# Input 1:
{
  "business_name": "Mama's Pizzeria",
  "rating": 5,
  "text": "The pizza here is the best I've ever had! The staff was also very friendly, I will definitely come back again."
}
# Output 1:
{
  "label": "Valid",
  "reason": "The review describes a genuine dining experience at the location, and the 5-star rating is consistent with the positive text."
}
---
# Input 2:
{
  "business_name": "Burger Palace",
  "rating": 5,
  "text": "The best burger in town! Visit www.burgerpalacepromo.com now for a 20% discount!"
}
# Output 2:
{
  "label": "Advertisement",
  "reason": "The review contains a promotional external link, despite the 5-star rating."
}
---
# Input 3:
{
  "business_name": "The Grand Library Cafe",
  "rating": 3,
  "text": "My new phone takes really clear pictures. By the way, this place is way too noisy."
}
# Output 3:
{
  "label": "Irrelevant",
  "reason": "The main subject of the review is a new phone, which is unrelated to the cafe."
}
---
# Input 4:
{
  "business_name": "City Central Parking",
  "rating": 1,
  "text": "I've never been here, but I read online that the owner is very rude. I'll never go!"
}
# Output 4:
{
  "label": "Rant_Without_Visit",
  "reason": "The reviewer explicitly states they have never visited the location, and the 1-star rating reflects a strong negative sentiment based on hearsay."
}
---

# Task Instructions:
Now, strictly follow the above policies and examples to analyze and classify the following input JSON object.
Your output must be a single, valid JSON object, perfectly matching the format of the examples.

# Input JSON:
{{json_input_string}}

# Output JSON:
"""

load_dotenv()

df = pd.read_csv("data/reviews_with_places_1000_Illinois.csv")

results = []
failed_rows = []
client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def process_single_row(row_data, max_retries=2):
    """
    Processes a single row of data by sending it to an LLM API
    and parsing the response.

    Args:
        row_data (tuple): A tuple containing the index and the row (as a pandas Series).
        max_retries (int): Number of retries if the API call or parsing fails.

    Returns:
        dict or None: A dictionary with the processed result or None if an error occurs.
    """
    index, row = row_data

    review_data = {
        "business_name": row["name_y"],
        "rating": int(row["rating"]),
        "text": row["text"],
        "description": row["description"] if pd.notna(row["description"]) else "No description available",
        "category": row["category"] if pd.notna(row["category"]) else "No category available"
    }
    json_input_string = json.dumps(review_data, ensure_ascii=False)

    for attempt in range(max_retries):
        try:
            # 1. Make the API call
            completion = client.chat.completions.create(
                model="claude-sonnet-4-20250514",
                messages=[
                    {'role': 'system', 'content': ENGLSIH_SYS_PRMOPT}, 
                    {'role': 'user', 'content': json_input_string}
                ],
                response_format={"type": "json_object"},
                max_tokens=512
            )

            # 2. Parse the response
            raw_output = completion.choices[0].message.content.strip()
            if not raw_output:
                raise ValueError("空输出")

            llm_output = json.loads(raw_output)

            # 3. Format the result
            return {
                "business_name": row["name_y"],
                "text": row["text"],
                "predicted_label": llm_output.get("label"),
                "prediction_reason": llm_output.get("reason")
            }

        except Exception as e:
            print(f"Row {index + 1} attempt {attempt + 1} failed: {e}")

    # 如果所有尝试都失败，返回 None
    return {
        "index": index,
        "business_name": row["name_y"],
        "rating": row["rating"],
        "text": row["text"],
        "error": "Failed after retries"
    }


results = []
# Prepare tasks by creating a list of tuples, each containing an index and a row
tasks_to_run = list(df.iterrows())

# Set the number of concurrent workers. Adjust this based on your system and API rate limits.
CONCURRENCY = 10 

with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
    # Submit all tasks to the executor
    futures = [executor.submit(process_single_row, task) for task in tasks_to_run]

    # Process futures as they complete and show a progress bar
    for future in tqdm(as_completed(futures), total=len(tasks_to_run), desc="Processing reviews"):
        result = future.result()
        # Only append successful results to the list
        if result:
            results.append(result)

# Convert the list of results into a new DataFrame
results_df = pd.DataFrame(results)

# Save the results to a CSV file
results_df.to_csv("data/label_reviews_Illinois.csv", index=False)

df = pd.read_csv("data/label_reviews_Illinois.csv")
print("总共有", len(df), "条记录")
