import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')
nltk.download('wordnet')

# Small dataset
data = {
    'text': [
        "The Earth is flat and NASA lies",                # Fake
        "COVID-19 vaccine prevents severe illness",        # Real
        "Aliens built the pyramids in Egypt",              # Fake
        "Doctors use AI to detect cancer early",           # Real
    ],
    'label': ['FAKE', 'REAL', 'FAKE', 'REAL']
}

df = pd.DataFrame(data)

# Preprocessing function
def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove stopwords and lemmatize
    words = text.split()
    cleaned = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return ' '.join(cleaned)

# Apply preprocessing
df['clean_text'] = df['text'].apply(preprocess)

# Vectorize and train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

model = LogisticRegression()
model.fit(X, y)

# Prediction function
def predict_news(text):
    cleaned = preprocess(text)
    vect = vectorizer.transform([cleaned])
    return model.predict(vect)[0]

# Test with user input
if __name__ == "__main__":
    user_input = input("Enter a news sentence: ")
    prediction = predict_news(user_input)
    print(f"Prediction: {prediction}")
