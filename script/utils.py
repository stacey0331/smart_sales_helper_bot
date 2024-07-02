import re
from nltk.tokenize import TweetTokenizer
import numpy as np

tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)

def preprocess_text(text):
    text = re.sub(r'[^\w\s\']|[\d]', '', text)
    tokens = tokenizer.tokenize(text)
    return tokens

def sentences_to_embeddings(sentences, model, vector_size):
    embeddings = []
    for sentence in sentences:
        vector = np.zeros(vector_size)
        count = 0
        for word in sentence:
            if word in model:
                vector += model[word]
                count += 1
        if count != 0: # sentence/vector not all 0
            vector /= count
        embeddings.append(vector)
    return np.array(embeddings)