import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Import Support Vector Classifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from utils import preprocess_text, sentences_to_embeddings

def load_glove_model(glove_file):
    model = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            model[word] = embedding
    return model

glove_file = './script/glove.6B.300d.txt'
glove_model = load_glove_model(glove_file)

splits = {'train': 'train.csv', 'test': 'test.csv'}
train_path = "hf://datasets/osyvokon/pavlick-formality-scores/" + splits["train"]
test_path = "hf://datasets/osyvokon/pavlick-formality-scores/" + splits["test"]

# Read train and test datasets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

additional_data_path = "train_data.csv"
additional_df = pd.read_csv(additional_data_path)

# Combine datasets into a single DataFrame
df = pd.concat([train_df, test_df], ignore_index=True)
df = pd.concat([df, additional_df], ignore_index=True)
sentences = df['sentence'].tolist()
formality_scores = df['avg_score'].values

preprocessed_sentences = []
preprocessed_formality_scores = []
for sentence, score in zip(sentences, formality_scores):
    preprocessed_sentence = preprocess_text(sentence)
    if preprocessed_sentence:  # Ensure sentence is not empty after preprocessing
        preprocessed_sentences.append(preprocessed_sentence)
        preprocessed_formality_scores.append(1 if score <= 0 else 0)

preprocessed_formality_scores = np.array(preprocessed_formality_scores)

# Convert preprocessed sentences to GloVe embeddings
sentence_embeddings = sentences_to_embeddings(preprocessed_sentences, glove_model, vector_size=300)  # Assuming vector_size matches GloVe dimensions

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(sentence_embeddings, preprocessed_formality_scores, test_size=0.2, random_state=42)

svm_clf = SVC(probability=True)  # Initialize SVM with probability=True for predict_proba support
svm_clf.fit(X_train, Y_train)
y_pred_proba = svm_clf.predict_proba(X_test)[:, 1]
y_pred = svm_clf.predict(X_test)

joblib.dump(svm_clf, './model/svm_model.pkl')
joblib.dump(glove_model, './model/glove_model.pkl')

# Evaluate model performance
accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(Y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(Y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

