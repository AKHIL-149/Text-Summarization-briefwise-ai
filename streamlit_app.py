import streamlit as st
import nltk
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

# Download NLTK punkt tokenizer
nltk.download("punkt")

# ------------------------------------------------
# Load Models Once (slow, so do it outside UI logic)
# ------------------------------------------------

@st.cache_resource
def load_models():
    models = {}

    # BART
    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    models["bart"] = {"model": bart_model, "tokenizer": bart_tokenizer}

    # T5
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    models["t5"] = {"model": t5_model, "tokenizer": t5_tokenizer}

    # Pegasus
    pegasus_model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
    pegasus_tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
    models["pegasus"] = {"model": pegasus_model, "tokenizer": pegasus_tokenizer}

    return models

@st.cache_resource
def load_sentence_bert():
    return SentenceTransformer("all-MiniLM-L6-v2")

models = load_models()
bert_model = load_sentence_bert()

# ROUGE scorer
rouge_scorer_obj = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

# ------------------------------------------------
# Functions
# ------------------------------------------------

def generate_summary(model_name, text):
    tokenizer = models[model_name]["tokenizer"]
    model = models[model_name]["model"]

    if model_name == "t5":
        text = f"summarize: {text}"

    inputs = tokenizer.encode(
        text, return_tensors="pt", max_length=1024, truncation=True
    )
    summary_ids = model.generate(
        inputs,
        max_length=50,
        min_length=20,
        num_beams=7,
        repetition_penalty=1.2,
        early_stopping=True,
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

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
        "bart_t5": round(
            util.pytorch_cos_sim(embeddings["bart"], embeddings["t5"]).item(), 4
        ),
        "bart_pegasus": round(
            util.pytorch_cos_sim(embeddings["bart"], embeddings["pegasus"]).item(), 4
        ),
        "t5_pegasus": round(
            util.pytorch_cos_sim(embeddings["t5"], embeddings["pegasus"]).item(), 4
        ),
    }
    return bert_similarities

# ------------------------------------------------
# Streamlit UI
# ------------------------------------------------

st.set_page_config(page_title="Multi-Model Text Summarization", layout="wide")

st.title("Multi-Model Text Summarization Evaluation")

# --------------------------
# INPUT TEXT + BUTTONS
# --------------------------

st.subheader("Input Text")

text_input = st.text_area(
    "Enter your text here...",
    height=200,
    label_visibility="collapsed"
)

col_btn1, col_btn2 = st.columns([1, 1])

with col_btn1:
    submit = st.button("Summarize", use_container_width=True)

with col_btn2:
    clear = st.button("Clear", use_container_width=True)

if clear:
    st.experimental_rerun()

# --------------------------
# Initialize placeholders
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

# --------------------------
# Summarization Processing
# --------------------------

if submit and text_input.strip():
    with st.spinner("Generating summaries..."):
        summaries = {
            "bart": generate_summary("bart", text_input),
            "t5": generate_summary("t5", text_input),
            "pegasus": generate_summary("pegasus", text_input),
        }

        bart_summary = summaries["bart"]
        t5_summary = summaries["t5"]
        pegasus_summary = summaries["pegasus"]

        rouge_scores = {
            model: calculate_rouge(text_input, summary)
            for model, summary in summaries.items()
        }

        bart_rouge = rouge_scores["bart"]
        t5_rouge = rouge_scores["t5"]
        pegasus_rouge = rouge_scores["pegasus"]

        similarity_tfidf = calculate_similarity_tfidf(summaries)
        similarity_bert = calculate_similarity_bert(summaries)

# --------------------------
# DISPLAY RESULTS
# --------------------------

st.markdown("---")
st.subheader("BART Summary")
st.info(bart_summary)
st.write(f"**ROUGE Score:** {bart_rouge}")

st.markdown("---")
st.subheader("T5 Summary")
st.info(t5_summary)
st.write(f"**ROUGE Score:** {t5_rouge}")

st.markdown("---")
st.subheader("Pegasus Summary")
st.info(pegasus_summary)
st.write(f"**ROUGE Score:** {pegasus_rouge}")

st.markdown("---")
st.subheader("TF-IDF Scores")
st.write(f"BART vs T5: **{similarity_tfidf['bart_t5']}**")
st.write(f"BART vs Pegasus: **{similarity_tfidf['bart_pegasus']}**")
st.write(f"T5 vs Pegasus: **{similarity_tfidf['t5_pegasus']}**")

st.markdown("---")
st.subheader("Sentence - BERT Scores")
st.write(f"BART vs T5: **{similarity_bert['bart_t5']}**")
st.write(f"BART vs Pegasus: **{similarity_bert['bart_pegasus']}**")
st.write(f"T5 vs Pegasus: **{similarity_bert['t5_pegasus']}**")
