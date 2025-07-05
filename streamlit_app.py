import streamlit as st
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
)
from evaluate import load
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# Download NLTK data
nltk.download("punkt")

# Load ROUGE metric
rouge_metric = load("rouge")

# Load BERT Sentence Transformer for semantic similarity
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load all summarization models
@st.cache_resource
def load_models():
    models = {
        "bart": {
            "model": BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn"),
            "tokenizer": BartTokenizer.from_pretrained("facebook/bart-large-cnn"),
        },
        "t5": {
            "model": T5ForConditionalGeneration.from_pretrained("t5-small"),
            "tokenizer": T5Tokenizer.from_pretrained("t5-small"),
        },
        "pegasus": {
            "model": PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum"),
            "tokenizer": PegasusTokenizer.from_pretrained("google/pegasus-xsum"),
        },
    }
    return models

models = load_models()

# Function to generate summary
def generate_summary(model_name, text):
    tokenizer = models[model_name]["tokenizer"]
    model = models[model_name]["model"]

    inputs = tokenizer.encode(
        text, return_tensors="pt", max_length=1024, truncation=True
    )
    if len(inputs[0]) == 0:
        return "Error: Input too short to summarize."

    summary_ids = model.generate(
        inputs,
        max_length=50,
        min_length=20,
        num_beams=7,
        repetition_penalty=1.2,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Function to calculate ROUGE score
def calculate_rouge(reference, generated_summary):
    scores = rouge_metric.compute(
        predictions=[generated_summary],
        references=[reference]
    )
    return round(float(scores["rouge1"]), 4)

# TF-IDF similarity
def calculate_similarity_tfidf(summaries):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(list(summaries.values()))
    cosine_similarities = cosine_similarity(tfidf_matrix)
    return {
        "bart_t5": round(cosine_similarities[0, 1], 4),
        "bart_pegasus": round(cosine_similarities[0, 2], 4),
        "t5_pegasus": round(cosine_similarities[1, 2], 4),
    }

# BERT-based similarity
def calculate_similarity_bert(summaries):
    embeddings = {model: bert_model.encode(summary, convert_to_tensor=True)
                  for model, summary in summaries.items()}
    similarities = {
        "bart_t5": round(util.pytorch_cos_sim(embeddings["bart"], embeddings["t5"]).item(), 4),
        "bart_pegasus": round(util.pytorch_cos_sim(embeddings["bart"], embeddings["pegasus"]).item(), 4),
        "t5_pegasus": round(util.pytorch_cos_sim(embeddings["t5"], embeddings["pegasus"]).item(), 4),
    }
    return similarities

# --- STREAMLIT UI ---

st.set_page_config(layout="wide")
st.title("Multi-Model Text Summarization Evaluation")

# Layout columns
col1, col2, col3 = st.columns([1.5, 1.5, 1.0])

# --- Input Box ---
with col1:
    st.subheader("Input Text")
    text_input = st.text_area("Enter your text here...", height=200)
    summarize_clicked = st.button("Summarize")
    if st.button("Clear"):
        st.experimental_rerun()

# Default summaries
summaries = {
    "bart": "Your summary will appear here...",
    "t5": "Your summary will appear here...",
    "pegasus": "Your summary will appear here...",
}

rouge_scores = {
    "bart": "-",
    "t5": "-",
    "pegasus": "-",
}

similarity_tfidf = {
    "bart_t5": "-",
    "bart_pegasus": "-",
    "t5_pegasus": "-",
}

similarity_bert = {
    "bart_t5": "-",
    "bart_pegasus": "-",
    "t5_pegasus": "-",
}

if summarize_clicked and text_input.strip():
    # Generate summaries
    summaries = {model: generate_summary(model, text_input) for model in models}

    # Compute ROUGE
    rouge_scores = {
        model: calculate_rouge(text_input, summaries[model])
        for model in models
    }

    # Compute similarities
    similarity_tfidf = calculate_similarity_tfidf(summaries)
    similarity_bert = calculate_similarity_bert(summaries)

# --- Output Boxes ---
with col2:
    st.subheader("BART Summary")
    st.write(summaries["bart"])
    st.markdown(f"**ROUGE Score:** {rouge_scores['bart']}")

    st.subheader("T5 Summary")
    st.write(summaries["t5"])
    st.markdown(f"**ROUGE Score:** {rouge_scores['t5']}")

    st.subheader("Pegasus Summary")
    st.write(summaries["pegasus"])
    st.markdown(f"**ROUGE Score:** {rouge_scores['pegasus']}")

# --- Similarity Scores ---
with col3:
    st.subheader("TF-IDF Scores", divider="grey")
    st.write(f"BART vs T5: {similarity_tfidf['bart_t5']}")
    st.write(f"BART vs Pegasus: {similarity_tfidf['bart_pegasus']}")
    st.write(f"T5 vs Pegasus: {similarity_tfidf['t5_pegasus']}")

    st.subheader("Sentence - BERT Scores", divider="grey")
    st.write(f"BART vs T5: {similarity_bert['bart_t5']}")
    st.write(f"BART vs Pegasus: {similarity_bert['bart_pegasus']}")
    st.write(f"T5 vs Pegasus: {similarity_bert['t5_pegasus']}")
