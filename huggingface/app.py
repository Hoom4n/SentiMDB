import gradio as gr
import nltk
import joblib

nltk.download("punkt")
nltk.download("wordnet")
nltk.download('punkt_tab')

model = joblib.load("sentiment_pipeline.joblib")

def predict_sentiment(text, show_preprocessed=False):
    proba = model.predict_proba([text])[0]
    sentiment = "Positive 😀" if proba[1] >= 0.5 else "Negative 😞"
    confidence = f"{round(max(proba) * 100, 2)}%"
    pre_out = ""
    if show_preprocessed:
        pre_out = model.named_steps["textpreprocessor"].transform([text])[0]
    return sentiment, confidence, pre_out

with gr.Blocks(css="""
.pipeline-container {
    background-color: blue;
    border:1px solid #ddd;
    border-radius:8px;
    padding:8px;
    margin-bottom: 4px;
}
.footer {
    margin-top: 24px;
    font-size:0.9rem;
    text-align:center;
}
""") as demo:

    gr.Markdown("# 🎬 SentiMDB")
    gr.Markdown(
        "### SentiMDB is a lightweight, production-ready Sentiment Analysis Pipeline based on IMDb movie reviews. It features a Flask web app, a Dockerized setup for easy deployment, and a Hugging Face Spaces-powered online demo. The project includes a comprehensive Jupyter Notebook, offering a guide to English Text Preprocessing and detailing the full Machine Learning Development process, including Model Selection, Error Analysis, and Fine-Tuning. By leveraging classic machine learning tools alone, the model achieved 91.67% prediction accuracy."
    )
  

    with gr.Row():
        with gr.Column(scale=1):

          
            gr.HTML("""
            <div class="pipeline-container">
              <h4 style="text-align:center; margin:0 0 8px 0;">Pipeline</h4>
              <div style="display:flex; justify-content:space-around; align-items:center;">
                <div>📝 Input Text</div>
                <div>→</div>
                <div>🔧 TextPreprocessor</div>
                <div>→</div>
                <div>📊 TF‑IDF</div>
                <div>→</div>
                <div>🤖 Logistic Regressor</div>
              </div>
            </div>
            """)

            
            
            review = gr.Textbox(lines=3, placeholder="Type your movie review here…")
            show_pre = gr.Checkbox(label="Show Preprocessed text", value=True)
            analyze_btn = gr.Button("Analyze", variant="primary")

            

        with gr.Column(scale=1):
           
            sentiment_out = gr.Label(label="Sentiment")
            confidence_out = gr.Textbox(label="Confidence")
            pre_out = gr.Textbox(label="Preprocessed Text", interactive=False)

            

    
    analyze_btn.click(
        fn=predict_sentiment,
        inputs=[review, show_pre],
        outputs=[sentiment_out, confidence_out, pre_out]
    )

    
    gr.HTML(
        '<div class="footer">For the full project Jupyter Notebook, Flask Web App & Docker Config, visit: <a href="https://github.com/Hoom4n/SentiMDB" target="_blank">https://github.com/Hoom4n/SentiMDB</a></div>'
    )

if __name__ == "__main__":
    demo.launch()