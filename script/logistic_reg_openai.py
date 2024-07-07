import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from utils import preprocess_text, sentences_to_embeddings

def process_embedding_string(embedding_list):
    float_embeddings = []
    for embedding in embedding_list:
        str_list = embedding[1:len(embedding)-1].split(',')
        float_embeddings.append([float(i) for i in str_list])
    return float_embeddings

if __name__ == '__main__':

    # splits = {'train': 'train.csv', 'test': 'test.csv'}
    # train_path = "hf://datasets/osyvokon/pavlick-formality-scores/" + splits["train"]
    # test_path = "hf://datasets/osyvokon/pavlick-formality-scores/" + splits["test"]

    # Read train and test datasets
    train_df = pd.read_csv("pavlick_train.csv")
    test_df = pd.read_csv("pavlick_test.csv")

    # Combine train and test datasets into a single DataFrame
    df = pd.concat([train_df, test_df], ignore_index=True)
    sentences = df['embedding'].tolist()
    formality_scores = df['avg_score'].values

    preprocessed_sentences = []
    preprocessed_formality_scores = []
    for score in formality_scores:
        preprocessed_formality_scores.append(1 if score <= 0 else 0)

    preprocessed_formality_scores = np.array(preprocessed_formality_scores)

    # Convert embedding string to list
    sentence_embeddings = process_embedding_string(sentences)

    X_train, X_test, Y_train, Y_test = train_test_split(sentence_embeddings, preprocessed_formality_scores, test_size=0.2, random_state=42)

    print("start fitting...")
    
    # Initialize and train a logistic regression model
    log_reg = LogisticRegression()
    log_reg.fit(X_train, Y_train)
    y_pred_proba = log_reg.predict_proba(X_test)[:, 1]
    y_pred = log_reg.predict(X_test) 

    joblib.dump(log_reg, '.logistic_reg_model.pkl')
    # joblib.dump(glove_model, './model/glove_model.pkl')

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
