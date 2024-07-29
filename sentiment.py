import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_csv('social_media_sentiments.csv')

# Split the dataset into features and labels
X = df['text']
y = df['sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a pipeline with a TfidfVectorizer and an SVM classifier
pipeline = make_pipeline(
    TfidfVectorizer(stop_words='english'),
    SVC(kernel='linear', probability=True)
)

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Example usage: Predict sentiment for new texts
def predict_sentiment(texts):
    predictions = pipeline.predict(texts)
    return predictions

# Test with some example texts
example_texts = [
    "I absolutely love this product!",
    "This is the worst experience I have ever had.",
    "It's okay, nothing special.",
]

predictions = predict_sentiment(example_texts)
for text, prediction in zip(example_texts, predictions):
    print(f"Text: {text}\nPredicted Sentiment: {prediction}\n")
