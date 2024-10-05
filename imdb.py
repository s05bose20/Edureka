# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.datasets import fetch_openml

# 1. Load the IMDB movie reviews dataset (positive/negative sentiment)
imdb_data = fetch_openml('imdb', version=1)
X = imdb_data.data  # Reviews (text data)
y = imdb_data.target  # Labels (positive or negative)

# 2. Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Preprocess the text data using CountVectorizer (convert text to feature vectors)
vectorizer = CountVectorizer(stop_words='english', max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Train a Logistic Regression classifier on the vectorized text data
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_vec, y_train)

# 5. Predict the sentiment of the test data
y_pred = lr.predict(X_test_vec)

# 6. Evaluate the model using accuracy, precision, and recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='positive')
recall = recall_score(y_test, y_pred, pos_label='positive')

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
