# Multi-Model Text Summarization

A web application and API for generating and evaluating text summaries using state-of-the-art transformer models: **BART**, **T5**, and **Pegasus**. The project provides both a Flask API, a Streamlit dashboard, and a static HTML/JS frontend for interactive experimentation and comparison of different summarization models.

---

## Features

- **Multiple Summarization Models**: Supports BART, T5, and Pegasus via HuggingFace Transformers.
- **Evaluation Metrics**: Calculates ROUGE-1 F1 scores for each summary.
- **Similarity Analysis**: Compares model outputs using both TF-IDF cosine similarity and Sentence-BERT embeddings.
- **User Interfaces**:
  - Streamlit web app for local or cloud usage.
  - Flask API for programmatic access.
  - Static HTML/JS frontend for basic web interaction.

---

## Project Structure

```
.
├── app.py              # Flask API backend
├── streamlit_app.py    # Streamlit dashboard frontend
├── index.html          # Static HTML frontend
├── styles.css          # Styles for HTML frontend
├── app.js              # JS logic for HTML frontend
├── requirements.txt    # Python dependencies
└── ...                 # Other files
```

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Yaswanth-pati/Text-Summarization.git
   cd Text-Summarization
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Main dependencies:
   - `flask`, `flask-cors`
   - `streamlit`
   - `transformers`, `sentence-transformers`
   - `rouge-score`
   - `scikit-learn`
   - `nltk`

---

## Usage

### 1. Streamlit Web App

Launch the dashboard locally:
```bash
streamlit run streamlit_app.py
```
- Enter input text, click "Summarize", and view outputs and evaluation metrics.

### 2. Flask API

Start the API server:
```bash
python app.py
```
- The API will be available at `http://localhost:5100/api/summarize`.
- Example POST request:
    ```bash
    curl -X POST http://localhost:5100/api/summarize \
      -H "Content-Type: application/json" \
      -d '{"text": "Your input text here."}'
    ```

### 3. Static HTML Frontend

- Open `index.html` in your browser.
- Ensure the backend API (`app.py`) is running and accessible to the frontend.

---

## Example

Input:
```
Artificial Intelligence is a field of computer science focused on creating systems capable of performing tasks that typically require human intelligence...
```

Outputs:
- **BART summary:** "...summary text..."
- **T5 summary:** "...summary text..."
- **Pegasus summary:** "...summary text..."
- **ROUGE scores**, **TF-IDF**, and **BERT similarity** displayed for comparison.

---

## Evaluation

- ROUGE-1 F1 score is used to assess summarization quality.
- Pairwise similarity between model outputs is computed using:
  - TF-IDF vectorization + cosine similarity
  - Sentence-BERT embeddings + cosine similarity

---

## Notes

- No datasets are included; users provide their own input text.
- Running the models may require significant RAM/VRAM.
- Initial model downloads may take time.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [Streamlit](https://streamlit.io/)
- [ROUGE Score](https://github.com/google-research/text-metrics)
