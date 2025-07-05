import streamlit as st
import requests

# Set page config
st.set_page_config(page_title="Multi-Model Text Summarization", layout="wide")

# Apply some custom CSS for styling
st.markdown("""
    <style>
    .big-title {
        font-size:36px !important;
        font-weight: bold;
        margin-bottom: 25px;
    }
    .section-box {
        border: 2px solid #4CAF50;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 25px;
        background-color: #f9f9f9;
    }
    .summary-box {
        border: 1px solid #999;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        background-color: #ffffff;
    }
    .score {
        color: #007ACC;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ---- PAGE TITLE ----
st.markdown('<div class="big-title">Multi-Model Text Summarization Evaluation</div>', unsafe_allow_html=True)


# ==============
# INPUT SECTION
# ==============
with st.container():
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    
    st.subheader("Input Text")
    text_input = st.text_area("Enter your text here...", height=200, label_visibility="collapsed")
    
    col1, col2 = st.columns([1, 1])
    summarize_btn = col1.button("Summarize", type="primary")
    clear_btn = col2.button("Clear")

    if clear_btn:
        st.experimental_rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ============================
# OUTPUT SECTION (conditional)
# ============================
if summarize_btn and text_input.strip():
    # Call Flask API
    try:
        response = requests.post(
            "http://localhost:5100/api/summarize",
            json={"text": text_input.strip()}
        )
        response.raise_for_status()
        result = response.json()

        # Summaries
        summaries = result.get("summaries", {})
        rouge_scores = result.get("rouge_scores", {})
        similarity_tfidf = result.get("similarity_tfidf", {})
        similarity_bert = result.get("similarity_bert", {})

        with st.container():
            st.markdown('<div class="section-box">', unsafe_allow_html=True)
            
            st.subheader("Model Summaries")

            # Individual summary boxes
            for model_name in ["bart", "t5", "pegasus"]:
                st.markdown(f"""
                    <div class="summary-box">
                        <h3>{model_name.upper()} Summary</h3>
                        <p>{summaries.get(model_name, "No summary returned.")}</p>
                        <p class="score">ROUGE Score: {rouge_scores.get(model_name, "-")}</p>
                    </div>
                """, unsafe_allow_html=True)

            st.subheader("TF-IDF Similarity Scores")
            st.write(f"BART vs T5: {similarity_tfidf.get('bart_t5', '-')}")
            st.write(f"BART vs Pegasus: {similarity_tfidf.get('bart_pegasus', '-')}")
            st.write(f"T5 vs Pegasus: {similarity_tfidf.get('t5_pegasus', '-')}")

            st.subheader("Sentence-BERT Similarity Scores")
            st.write(f"BART vs T5: {similarity_bert.get('bart_t5', '-')}")
            st.write(f"BART vs Pegasus: {similarity_bert.get('bart_pegasus', '-')}")
            st.write(f"T5 vs Pegasus: {similarity_bert.get('t5_pegasus', '-')}")

            st.markdown('</div>', unsafe_allow_html=True)

    except requests.RequestException as e:
        st.error(f"API call failed: {e}")
else:
    if summarize_btn:
        st.warning("Please enter some text to summarize.")
