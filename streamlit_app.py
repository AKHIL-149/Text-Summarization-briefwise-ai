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
