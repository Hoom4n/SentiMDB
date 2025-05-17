# SentiMDB: Sentiment Analysis for IMDB Movie Reviews
  
SentiMDB is a lightweight, production-ready Sentiment Analysis Pipeline based on IMDb movie reviews. It features a Flask web app, a Dockerized setup for easy deployment, and a Hugging Face Spaces-powered online demo. The project includes a comprehensive <a href="https://nbviewer.org/github/hoom4n/SentiMDB/blob/main/notebook/SentiMDB.ipynb">Jupyter Notebook</a>, offering a guide to English Text Preprocessing and detailing the full Machine Learning Development process, including Model Selection, Error Analysis, and Fine-Tuning. By leveraging classic machine learning tools alone, the model achieved 91.67% prediction accuracy.

![SentiMDB Home](images/senimdb_home.png)


## ✅ Features 

- **End-to-End Lightweight Sentiment Analysis Pipeline**  
  A complete and efficient sentiment analysis solution, with the entire pipeline packaged in just 1.7MB.
- **Flexible and Reusable TextPreprocessor Transformer**  
  Developed in this project and available on <a href="https://pypi.org/project/hoomanmltk/">PyPI</a>, for easy reuse.
- **Model Optimization Techniques** 
  Error analysis, Hyperparameter tuning with Optuna.
- **Production-Ready Deployment**  
  Streamlit Online Demo, Flask web app, Dockerized.

## 🛠️ Installation 

### Option 1: Online Demo

You can try the online demo on HuggingFace Spaces: <a href="https://huggingface.co/spaces/hoom4n/SentiMDB>https://huggingface.co/spaces/hoom4n/SentiMDB</a>

### Option 2: Local Setup

```bash
# clone project repository
git clone https://github.com/yourusername/SentiMDB.git
cd SentiMDB

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# venv\Scripts\activate  # On Windows

pip install -r requirements.txt

# Run Flask application
python app/app.py
```

Access applicaion at: http://localhost:5000

### Option 3: Docker Deployment

```bash
# clone project repository
git clone https://github.com/yourusername/SentiMDB.git
cd SentiMDB

# Build image and start container for the first time
docker compose up --build

# For subsequent runs
docker compose up
```

Access applicaion at: http://localhost:8080

## 🧠 Machine Learning Development Process 

### 📊 EDA & Text Preprocessing Guide  
A quick exploratory data analysis (EDA) was performed, including analyzing word length distribution, target class balance, and other key dataset characteristics.  
The Notebook also provides a Text Preprocessing Guide, demonstrating steps like Text Cleaning, Tokenization, and Normalization, with Feature Extraction using Bag of Words (BoW) and TF-IDF. The guide includes both theory and an example applied to a single dataset sample.

### 🧰 Reusable and Flexible TextProcessor Transformer 
Developed a customizable TextPreprocessor to seamlessly integrate into Scikit-learn pipelines. It supports parallel processing for faster execution, utilizing Joblib to leverage multiple CPU cores. The class offers flexible tuning, input validation, and logging for preprocessing errors. It’s reusable, available on <a href="https://pypi.org/project/hoomanmltk/">PyPI</a>, and can be found on GitHub (<a href="https://github.com/Hoom4n/HoomanMLTK">HoomanMLTK</a>).

### 🧪 Model Selection 
Several classification algorithms suitable for text classification tasks were evaluated using 5-fold cross-validation, with essential text preprocessing and default model settings. **Logistic Regression** outperformed the other classifiers.

<div align="center">
  
| Model                     | Train F1 | CV F1   | CV F1 STD | Runtime (s) |
|---------------------------|----------|---------|-----------|-------------|
| Logistic Reg.             | 0.974    | 0.886   | 0.002817  | 93.4        |
| Linear SVC                | 1.000    | 0.870   | 0.002235  | 96.6        |
| LightGBM                  | 0.897    | 0.861   | 0.001084  | 97.3        |
| XGBoost                   | 0.935    | 0.860   | 0.001935  | 107.5       |
| Random Forest             | 1.000    | 0.848   | 0.000990  | 195.4       |
| Multinomial NB            | 0.899    | 0.843   | 0.002726  | 90.8        |
| AdaBoost                  | 0.810    | 0.807   | 0.006110  | 114.4       |
| K-Nearest Neighbors       | 0.781    | 0.663   | 0.003233  | 210.8       |

</div>

### 🧐 Error Analysis  
Logistic Regression was chosen for further improvement and was analyzed with the following methods:  

  - **Performance and Confusion Matrix**: The confusion matrix revealed a balance error between false negatives and false positives, with an F1 score of 0.89 and an accuracy of 89%.  
  - **Analysis of Misclassified Samples**: By manually inspecting some of the false positives and false negatives, it was found that most misclassifications were caused by sentiment shifts, mixed sentiments, and ambiguous reviews. Additionally, better handling of negations (e.g., “don’t” → “do not”) and the inclusion of 2-grams were suggested as improvements to capture more context in the text, which could help reduce misclassifications.  
  - **Learning Curve**: The learning curve indicated that Logistic Regression was overfitting, highlighting the need for regularization to constrain the model and improve generalization.  
  - **Word Importance**:
Most impactful words influencing the classifier's sentiment predictions were identified. 

<div align="center">
  
<img src="images/word_impotances.png" alt="Word Importance" width="85%"/>

</div>

### 🎯 Fine-Tuning  

  - **Fine-tuning TextProcessor with Optuna**: Based on insights from the Error Analysis and the tunability of the TextProcessor, several techniques were evaluated using Optuna. These included stemming, various vectorizers (CountVectorizer and TF-IDF), expanding contractions, including bigrams and unigrams and apply some vectorizer limitations such as restricting the vocabulary size.  
  - **Fine-tuning the Classifier**: A separate Optuna study was conducted to fine-tune the Logistic Regression model, exploring a search space to apply constraints and regularization techniques (e.g., L1/L2 regularizers, solver selection, early stopping tolerance). These optimizations aimed to improve the model's generalization power.  
  - **Results**: By fine-tuning both the preprocessing pipeline and the classifier, a 3% reduction in false negatives and a 2% reduction in false positives were achieved compared to the baseline model. This fine-tuning led to an 18% decrease in overall error, improving the validation F1 score and accuracy to 91%.  
  - **Sentiment Lexicon Ensembling**  
To improve FN/FP predictions, a sentiment lexicon-based approach was developed using VADER. A custom transformer was implemented to extract VADER scores from raw text, which were used as input to an LGBM classifier, with dataset labels as output. Predictions were combined with a tuned logistic regression model via Soft Voting and Stacking. As performance did not improve, the approach was excluded to avoid unnecessary pipeline complexity.

### 📈 Model Generalization on Final Test Set
The final tuned pipeline was evaluated on a test set that was kept separate throughout the project. It achieved an accuracy of 91.67%, indicating the model's strong generalization capabilities.

### 💾 Save for Deployment  
The final tuned pipeline was saved as a joblib file for easy deployment.

## 🚀 Future Work 

- **Advanced Text Preprocessing Techniques**  
Using more sophisticated text preprocessing techniques—such as word embeddings (e.g., Word2Vec, GloVe) or contextual models like BERT—will help capture deeper linguistic nuances, including sarcasm, irony, sentiment shifts, mixed sentiments, and ambiguity in reviews.

- **Enhanced Modeling Approaches**  
Neural network-based classifiers—such as RNNs, CNNs, or transformers—will be employed to better capture complex sentiment patterns and contextual dependencies beyond the capabilities of traditional models.

- **Broader and Multi-Domain Datasets**  
The model will be trained on a more diverse set of datasets spanning multiple domains, including movie reviews, social media posts, and customer feedback, to enhance generalizability and robustness across varied contexts.
