from bertopic import BERTopic
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer

import numpy

print(f"Version: {numpy.__version__}")

data = load_dataset("HuggingFaceH4/h4_10k_prompts_ranked_gen")
docs = data["train_gen"]["prompt"]

vectorizer_model = CountVectorizer(stop_words="english")

seed_topic_list = [
    ["when", "date", "time"],
    ["who"],
    ["where"],
    ["how"],
    ["rephrase", "reword"],
    ["translate"],
    ["extract"],
    ["code", "coding", "python"],
    ["imagine", "act", "assume", "role"],
]
# searching knowledge', 'answer coding problem', 'summarizing', 'rephrasing', 'roleplay', 'translate', 'generate content

topic_model = BERTopic(
    seed_topic_list=seed_topic_list,
    vectorizer_model=vectorizer_model,
    min_topic_size=25,
)

topics, probs = topic_model.fit_transform(docs)
topic_model.get_topic_info()
