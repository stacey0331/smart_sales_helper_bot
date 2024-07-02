import pandas as pd
import re
from nltk.tokenize import TweetTokenizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_glove_model(glove_file):
    model = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            model[word] = embedding
    return model

glove_file = './script/glove.6B.300d.txt'  # Path to GloVe file
glove_model = load_glove_model(glove_file)

tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)

def preprocess_text(text):
    text = re.sub(r'[^\w\s\']|[\d]', '', text)  # Remove special characters, punctuation, and digits
    tokens = tokenizer.tokenize(text)  # Tokenize text into words
    return tokens

splits = {'train': 'train.csv', 'test': 'test.csv'}
df = pd.read_csv("hf://datasets/osyvokon/pavlick-formality-scores/" + splits["train"])
sentences = df['sentence'].tolist()
formality_scores = df['avg_score'].values

preprocessed_sentences = []
filtered_formality_scores = []
for sentence, score in zip(sentences, formality_scores):
    preprocessed_sentence = preprocess_text(sentence)
    if preprocessed_sentence:  # Ensure sentence is not empty after preprocessing
        preprocessed_sentences.append(preprocessed_sentence)
        filtered_formality_scores.append(score)

# Convert formality scores to numpy array
filtered_formality_scores = np.array(filtered_formality_scores)

# Function to convert sentences to GloVe embeddings
def sentences_to_embeddings(sentences, model, vector_size):
    embeddings = []
    for sentence in sentences:
        vector = np.zeros(vector_size)
        count = 0
        for word in sentence:
            if word in model:
                vector += model[word]
                count += 1
        if count != 0:
            vector /= count
        else:
            print(f"Words not found in GloVe model: {sentence}")
        embeddings.append(vector)
    return np.array(embeddings)

# Convert preprocessed sentences to GloVe embeddings
sentence_embeddings = sentences_to_embeddings(preprocessed_sentences, glove_model, vector_size=300)  # Assuming vector_size matches GloVe dimensions

# Check for any zero vectors
for i, embedding in enumerate(sentence_embeddings):
    if not np.any(embedding):  # Check if the vector is all zeros
        print(f"Zero vector found for sentence index {i}")

X_train, X_test, Y_train, Y_test = train_test_split(sentence_embeddings, filtered_formality_scores, test_size=0.2, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train, Y_train)

y_pred = reg_model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)
print(f"Mean Squared Error: {mse}")     # Lower MSE indicates better model performance
print(f"R^2 Score: {r2}")               # higher == arger proportion of the variance in the target variable can be explained by the independent variables

# Calculate residuals
residuals = Y_test - y_pred

# Plot histogram of residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Histogram of Residuals')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()

# Scatter plot of residuals vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs. Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()
