from flask import Flask, render_template, request
from joblib import load
import nltk
import os

app = Flask(__name__)

# Load pipeline
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'sentiment_pipeline.joblib')
model = load(model_path)

@app.route('/', methods=['GET', 'POST'])
def home():
    preprocessed = None
    show_pre = False

    if request.method == 'POST':
        text = request.form['text']
        show_pre = 'show_preprocessed' in request.form

        # predict
        prediction = model.predict_proba([text])
        if prediction[0][1] >= prediction[0][0]:
            sentiment = "Positive"
            confidence = round(prediction[0][1] * 100, 2)
        else:
            sentiment = "Negative"
            confidence = round(prediction[0][0] * 100, 2)

        # get preprocessed text if requested
        if show_pre:
            preprocessed = model.named_steps['textpreprocessor'] \
                                .transform([text])[0]

        return render_template(
            'index.html',
            result=True,
            text=text,
            sentiment=sentiment,
            confidence=confidence,
            show_pre=show_pre,
            preprocessed=preprocessed
        )

    return render_template('index.html', result=False, show_pre=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
