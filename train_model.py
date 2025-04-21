import pandas as pd
import nltk
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

nltk.download('stopwords')

# Load fake and real news (updated paths to 'data' folder)
df_fake = pd.read_csv('data/Fake.csv')
df_real = pd.read_csv('data/True.csv')

# Add labels
df_fake['label'] = 0  # FAKE
df_real['label'] = 1  # REAL

# Combine datasets
df = pd.concat([df_fake, df_real])
df = df[['text', 'label']]

# Shuffle dataset
df = df.sample(frac=1).reset_index(drop=True)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the classifier
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_vec, y_train)

# Evaluate the model
preds = model.predict(X_test_vec)
acc = accuracy_score(y_test, preds)
print(f"Model accuracy: {acc * 100:.2f}%")

# Save model and vectorizer
with open('fake_news_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
