# MailGuard — Spam Email Classifier

A simple supervised NLP application that classifies a message as spam or not spam.

## Features
- Paste an email, SMS, or message for analysis.
- Binary spam / not-spam prediction.
- TF-IDF feature extraction using unigrams and bigrams.
- Multinomial Naive Bayes classifier.
- Model test accuracy, classification report, and confusion matrix.
- Educational explanation inside the app.

## Technology
- Python
- Streamlit
- Pandas
- Scikit-learn
- Natural Language Processing (NLP)

## Run locally
1. Install Python 3.10 or newer.
2. Open this folder in VS Code.
3. Create and activate a virtual environment:

   Windows PowerShell:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Start the app:
   ```bash
   streamlit run app.py
   ```

The browser will open the application, usually at http://localhost:8501.

## Dataset
`messages.csv` contains a small balanced educational dataset with `message` and `label` columns. `ham` means legitimate message; `spam` means unwanted or suspicious message.

## Algorithm
The app uses a Scikit-learn Pipeline:
- `TfidfVectorizer` converts message text into numerical word and bigram features.
- `MultinomialNB` learns word-feature patterns for the spam and ham classes.
- A stratified train/test split is used to report basic evaluation metrics.

## Limitations
The included dataset is intentionally small and handcrafted for a runnable classroom demonstration. Its accuracy is not representative of a production spam filter. For stronger real-world performance, train and evaluate on a larger, public, properly licensed dataset such as the SMS Spam Collection, and test on messages from a separate source. The prediction is not a security guarantee.
