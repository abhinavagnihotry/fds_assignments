import os
import re
import nltk
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LdaModel
from pprint import pprint
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('stopwords')
folder_path = os.path.expanduser('~/Downloads/UN_data/TXT')
processed_speeches = []
stop_words = set(stopwords.words('english'))

nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()
country_sentiments = {}

for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.txt') and not file.startswith("._"):
            # Assuming country name is the first part of the file name, e.g., 'COM_77_2022.txt'
            country_name = file.split('_')[0]

            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

                # Perform sentiment analysis on the content
                sentiment_scores = sia.polarity_scores(content)

                # Store the compound score for each country
                country_sentiments[country_name] = sentiment_scores['compound']

# Compute the mean compound score
mean_compound_score = sum(country_sentiments.values()) / len(country_sentiments)

# Print the compound score for each country and the mean score
for country, compound in country_sentiments.items():
    print(f"Sentiment for {country}: {compound}")

print(f"\nMean Compound Sentiment Score: {mean_compound_score}")

def preprocess(text):

    text = re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z]', ' ', text)).lower()

    tokens = text.split()

    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return tokens


for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.txt'):
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                processed_speeches.append(preprocess(content))

dictionary = corpora.Dictionary(processed_speeches)
corpus = [dictionary.doc2bow(speech) for speech in processed_speeches]

lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, random_state=42, passes=15)

print("Themes/Topics in the speeches:")
pprint(lda_model.print_topics())

