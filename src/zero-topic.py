from bertopic import BERTopic
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer

data = load_dataset("HuggingFaceH4/h4_10k_prompts_ranked_gen")
docs = data["train_gen"]["prompt"]

zeroshot_topic_list = ['searching knowledge', 'answer coding problem', 'summarizing', 'rephrasing', 'roleplay', 'translate', 'generate content']
vectorizer_model = CountVectorizer(stop_words="english")

topic_model = BERTopic(
    min_topic_size=20,
    zeroshot_topic_list=zeroshot_topic_list,
    zeroshot_min_similarity=.25,
    vectorizer_model=vectorizer_model
)

topics, probs = topic_model.fit_transform(docs)
topic_model.get_topic_info()
