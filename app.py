import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="MailGuard | Spam Detector", page_icon="🛡️", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("messages.csv")

@st.cache_resource
def train_model():
    data = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        data["message"], data["label"], test_size=0.25,
        random_state=42, stratify=data["label"]
    )
    # Pipeline keeps text vectorization and classification together.
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
        ("classifier", MultinomialNB())
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=["ham", "spam"])
    return model, accuracy, report, matrix, len(data)

st.markdown("""
<style>
.main {background: linear-gradient(135deg,#101827,#17243a);}
.hero {padding:1.2rem 1.5rem;border-radius:18px;background:linear-gradient(100deg,#0f766e,#2563eb);color:white;}
.result {padding:1.2rem;border-radius:14px;background:#172033;border:1px solid #334155;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🛡️ MailGuard</h1><p>Check suspicious messages with a Natural Language Processing model.</p></div>', unsafe_allow_html=True)
st.write("Paste an email, SMS, or message below. The model will classify it as Spam or Not Spam.")

model, accuracy, report, matrix, total_rows = train_model()

col1, col2 = st.columns(2)
with col1:
    st.metric("Model test accuracy", f"{accuracy * 100:.1f}%")
with col2:
    st.metric("Training dataset size", f"{total_rows} messages")

message = st.text_area("Enter message text", height=180, placeholder="Example: Congratulations! You have won a free prize. Click here to claim...")
if st.button("🔍 Analyze message", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Please enter a message before analyzing.")
    else:
        prediction = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]
        classes = model.named_steps["classifier"].classes_
        confidence = float(probabilities[list(classes).index(prediction)]) * 100
        if prediction == "spam":
            st.error(f"🚨 Prediction: SPAM  •  Model confidence: {confidence:.1f}%")
            st.write("Be cautious. Avoid clicking links or sharing passwords, OTPs, or financial details.")
        else:
            st.success(f"✅ Prediction: NOT SPAM  •  Model confidence: {confidence:.1f}%")
            st.write("The model considers this message similar to legitimate examples.")
        st.caption("Confidence is the model's estimated class probability, not a guarantee that the message is safe.")

with st.expander("Model evaluation"):
    st.write("The model is evaluated on a held-out test split from the included sample dataset.")
    st.write("Confusion matrix (rows = actual; columns = predicted; order: ham, spam):")
    st.dataframe(pd.DataFrame(matrix, index=["Actual ham","Actual spam"], columns=["Predicted ham","Predicted spam"]))
    st.write("Classification report:")
    st.dataframe(pd.DataFrame(report).transpose().round(3))

with st.expander("How does the classifier work?"):
    st.markdown("""
    1. **Dataset:** Messages are labelled as `ham` (legitimate) or `spam`.
    2. **Train/test split:** The dataset is divided into training and testing portions.
    3. **TF-IDF:** Words and two-word phrases are converted into numerical features.
    4. **Multinomial Naive Bayes:** A probabilistic classifier learns which terms are associated with each class.
    5. **Prediction:** New text is transformed using the same TF-IDF vocabulary and classified.

    This is a supervised NLP classification project.
    """)

st.divider()
st.caption("Academic demonstration project • Python + Streamlit + Scikit-learn")
