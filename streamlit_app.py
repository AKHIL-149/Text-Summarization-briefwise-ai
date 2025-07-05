import streamlit as st
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
)
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import torch
import nltk

# Download NLTK punkt tokenizer
nltk.download("punkt")

st.set_page_config(page_title="Multi-Model Text Summarization", layout="wide")

st.title("Multi-Model Text Summarization Evaluation")

# ============================
# Load models ONCE (cached)
# ============================

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

@st.cache_resource
def load_bert():
    return SentenceTransformer("all-MiniLM-L6-v2")

models = load_models()
bert_model = load_bert()

rouge_scorer_obj = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

# ============================
# Summarization function
# ============================

def generate_summary(model_name, text):
    tokenizer = models[model_name]["tokenizer"]
    model = models[model_name]["model"]

    inputs = tokenizer.encode(
        text, return_tensors="pt", max_length=1024, truncation=True
    )

    if inputs.shape[1] == 0:
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

# ============================
# Metrics functions
# ============================

def calculate_rouge(reference, generated_summary):
    scores = rouge_scorer_obj.score(reference, generated_summary)
    rouge1_fmeasure = scores["rouge1"].fmeasure
    return round(float(rouge1_fmeasure), 4)

def calculate_similarity_tfidf(summaries):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(summaries.values())
    cosine_similarities = cosine_similarity(tfidf_matrix)
    
    return {
        "bart_t5": round(cosine_similarities[0, 1], 4),
        "bart_pegasus": round(cosine_similarities[0, 2], 4),
        "t5_pegasus": round(cosine_similarities[1, 2], 4),
    }

def calculate_similarity_bert(summaries):
    embeddings = {
        model: bert_model.encode(summary, convert_to_tensor=True)
        for model, summary in summaries.items()
    }
    bert_similarities = {
        "bart_t5": round(util.pytorch_cos_sim(embeddings["bart"], embeddings["t5"]).item(), 4),
        "bart_pegasus": round(util.pytorch_cos_sim(embeddings["bart"], embeddings["pegasus"]).item(), 4),
        "t5_pegasus": round(util.pytorch_cos_sim(embeddings["t5"], embeddings["pegasus"]).item(), 4),
    }
    return bert_similarities




import streamlit as st
import requests

st.set_page_config(page_title="Multi-Model Text Summarization", layout="wide")

st.title("Multi-Model Text Summarization Evaluation")

# Create 3 columns
col1, col2, col3 = st.columns([1, 1, 1])

# --------------------------
# LEFT COLUMN — INPUT BOX
# --------------------------

with col1:
    st.subheader("Input Text")
    
    # Text input box
    text_input = st.text_area(
        "Enter your text here...",
        height=200,
        label_visibility="collapsed"
    )
    
    # Buttons
    submit = st.button("Summarize", use_container_width=True)
    clear = st.button("Clear", use_container_width=True)
    
    if clear:
        st.experimental_rerun()

# --------------------------
# MIDDLE COLUMN — SUMMARIES
# --------------------------

bart_summary = "Your summary will appear here..."
t5_summary = "Your summary will appear here..."
pegasus_summary = "Your summary will appear here..."

bart_rouge = "-"
t5_rouge = "-"
pegasus_rouge = "-"

similarity_tfidf = {
    "bart_t5": "-",
    "bart_pegasus": "-",
    "t5_pegasus": "-"
}

similarity_bert = {
    "bart_t5": "-",
    "bart_pegasus": "-",
    "t5_pegasus": "-"
}

if submit and text_input.strip():
    # Call Flask API
    try:
        response = requests.post(
            "http://localhost:5100/api/summarize",  # CHANGE THIS for production
            json={"text": text_input},
            timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            
            bart_summary = data["summaries"]["bart"]
            t5_summary = data["summaries"]["t5"]
            pegasus_summary = data["summaries"]["pegasus"]
            
            bart_rouge = data["rouge_scores"]["bart"]
            t5_rouge = data["rouge_scores"]["t5"]
            pegasus_rouge = data["rouge_scores"]["pegasus"]
            
            similarity_tfidf = data["similarity_tfidf"]
            similarity_bert = data["similarity_bert"]
            
        else:
            st.error(f"API returned error: {response.text}")
    except Exception as e:
        st.error(f"API call failed: {e}")

with col2:
    st.subheader("BART Summary")
    st.write(bart_summary)
    st.write(f"**ROUGE Score:** {bart_rouge}")

    st.subheader("T5 Summary")
    st.write(t5_summary)
    st.write(f"**ROUGE Score:** {t5_rouge}")

    st.subheader("Pegasus Summary")
    st.write(pegasus_summary)
    st.write(f"**ROUGE Score:** {pegasus_rouge}")

# --------------------------
# RIGHT COLUMN — SCORES
# --------------------------

with col3:
    # TF-IDF
    st.subheader("TF-IDF Scores", divider="gray")
    st.write(f"BART vs T5: **{similarity_tfidf['bart_t5']}**")
    st.write(f"BART vs Pegasus: **{similarity_tfidf['bart_pegasus']}**")
    st.write(f"T5 vs Pegasus: **{similarity_tfidf['t5_pegasus']}**")
    
    st.markdown("---")
    
    # BERT
    st.subheader("Sentence - BERT Scores", divider="gray")
    st.write(f"BART vs T5: **{similarity_bert['bart_t5']}**")
    st.write(f"BART vs Pegasus: **{similarity_bert['bart_pegasus']}**")
    st.write(f"T5 vs Pegasus: **{similarity_bert['t5_pegasus']}**")
