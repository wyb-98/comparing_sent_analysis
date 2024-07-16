import nltk
import numpy as np

from tqdm import tqdm

from flair.models import TextClassifier
from flair.data import Sentence
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import DistilBertTokenizer

# Load the sentiment classifier
classifier = TextClassifier.load('sentiment-fast')

# VADER lexicon for text from social media
nltk.download('vader_lexicon')

# Initialize sentiment intensity analyzer for get_sent_ntlk
sia = SentimentIntensityAnalyzer()

def get_sent_flair(text):
    sentence = Sentence(text)
    classifier.predict(sentence)
    score = 0

    if sentence.labels[0].value == 'POSITIVE':
        score = 1

    return score

def get_sent_nltk(text):
    score = sia.polarity_scores(text)
    return score["compound"] > 0

def get_sent_textblob(text):
    blob = TextBlob(text)
    score = blob.sentiment[0]
    return score > 0

# DistilBert Tokenizer

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(data,max_len) :

    input_ids = []

    attention_masks = []

    for i in tqdm(range(len(data))):

        encoded = tokenizer.encode_plus(
            data[i],
            add_special_tokens=True,
            max_length = max_len,
            padding='max_length',
            return_attention_mask=True
        )

        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

        
    return np.array(input_ids),np.array(attention_masks)