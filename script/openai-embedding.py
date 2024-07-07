import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm

def get_embedding_ada(client, text):
    return client.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
        encoding_format="float"
    ).data[0].embedding
    
if __name__ == '__main__':
    
    # splits = {'train': 'train.csv', 'test': 'test.csv'}
    # train_path = "hf://datasets/osyvokon/pavlick-formality-scores/" + splits["train"]
    # test_path = "hf://datasets/osyvokon/pavlick-formality-scores/" + splits["test"]

    # Read train and test datasets
    train_df = pd.read_csv('pavlick_train.csv')
    test_df = pd.read_csv('pavlick_test.csv')
    
    envfile = open(".openaikey",'r')
    api_key = envfile.read()
    
    os.environ['OPENAI_API_KEY'] = api_key
    client = OpenAI()
    
    train_embedding, test_embedding =  [], []
    
    for i in tqdm(range(len(train_df))):
        sentence = train_df['sentence'][i]
        train_embedding.append(get_embedding_ada(client, sentence.lower()))
        
    train_df.insert(3, 'embedding', train_embedding, True)
    train_df.to_csv('pavlick_train.csv')  

    for i in tqdm(range(len(test_df))):
        sentence = test_df['sentence'][i]
        test_embedding.append(get_embedding_ada(client, sentence.lower()))
        
    test_df.insert(3, 'embedding', test_embedding, True)
    test_df.to_csv('pavlick_test.csv')  
    