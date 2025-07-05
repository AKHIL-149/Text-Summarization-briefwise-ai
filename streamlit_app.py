import streamlit as st
import requests

st.set_page_config(
    page_title="Multi-Model Text Summarization",
    layout="wide"
)

st.markdown("""
    <style>
        .btn-primary {
            background-color: #4CAF50;
            color: white;
            padding: 10px 16px;
            font-size: 16px;
            border: none;
            cursor: pointer;
            margin-right: 10px;
        }
        .btn-secondary {
            background-color: #f44336;
            color: white;
            padding: 10px 16px;
            font-size: 16px;
            border: none;
            cursor: pointer;
        }
        textarea {
            font-family: Arial, sans-serif;
            font-size: 16px;
        }
        .output-box {
            background-color: #f2f2f2;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Multi-Model Text Summarization Evaluation")

# Text Input
st.header("Input Text")
text = st.text_area(
    label="",
    height=200,
    placeholder="Enter your text here..."
)

col1, col2 = st.columns([1, 1])

summarize = col1.button("Summarize")
clear = col2.button("Clear")

if clear:
    st.experimental_rerun()

if summarize and text.strip():
    # Call Flask API
    response = requests.post(
        "http://localhost:5100/api/summarize",
        json={"text": text}
    )
    
    if response.status_code == 200:
        result = response.json()

        # Summaries Section
        st.header("Model Summaries")

        bart_summary = result["summaries"]["bart"]
        t5_summary = result["summaries"]["t5"]
        pegasus_summary = result["summaries"]["pegasus"]

        bart_score = result["rouge_scores"]["bart"]
        t5_score = result["rouge_scores"]["t5"]
        pegasus_score = result["rouge_scores"]["pegasus"]

        # Layout similar to HTML: 3 columns for summaries
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
                <div class="output-box">
                    <h3>BART Summary</h3>
                    <p>{bart_summary}</p>
                    <p class="score">ROUGE Score: <strong>{bart_score}</strong></p>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div class="output-box">
                    <h3>T5 Summary</h3>
                    <p>{t5_summary}</p>
                    <p class="score">ROUGE Score: <strong>{t5_score}</strong></p>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
                <div class="output-box">
                    <h3>Pegasus Summary</h3>
                    <p>{pegasus_summary}</p>
                    <p class="score">ROUGE Score: <strong>{pegasus_score}</strong></p>
                </div>
            """, unsafe_allow_html=True)

        st.divider()

        # TF-IDF Similarity
        st.header("TF-IDF Scores")
        sim_tfidf = result["similarity_tfidf"]
        st.markdown(f"""
            <p>BART vs T5: <strong>{sim_tfidf['bart_t5']}</strong></p>
            <p>BART vs Pegasus: <strong>{sim_tfidf['bart_pegasus']}</strong></p>
            <p>T5 vs Pegasus: <strong>{sim_tfidf['t5_pegasus']}</strong></p>
        """, unsafe_allow_html=True)

        st.divider()

        # BERT Similarity
        st.header("Sentence-BERT Scores")
        sim_bert = result["similarity_bert"]
        st.markdown(f"""
            <p>BART vs T5: <strong>{sim_bert['bart_t5']}</strong></p>
            <p>BART vs Pegasus: <strong>{sim_bert['bart_pegasus']}</strong></p>
            <p>T5 vs Pegasus: <strong>{sim_bert['t5_pegasus']}</strong></p>
        """, unsafe_allow_html=True)

    else:
        st.error(f"API error {response.status_code}: {response.text}")

elif summarize and not text.strip():
    st.warning("Please enter text before summarizing.")
