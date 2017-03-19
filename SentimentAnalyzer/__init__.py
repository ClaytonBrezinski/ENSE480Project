import sys
import pandas
from bs4 import BeautifulSoup as BS
import re
from nltk.corpus import stopwords # Import the stop word list
from sklearn.feature_extraction.text import CountVectorizer

def __init__():
    # prepare data for learning
    trainingIterations = 100
    trainingDataset = getPandasDataset('labeledTrainData.tsv', "\t")
    cleanedTrainingData = []
    for i in range(0, trainingDataset['review'].size):
        cleanedTrainingData.append(cleanDataset(trainingDataset['review'][i]))
        if (i+1) % 500 == 0:
            print("cleaned review %d" % (i+1) + " of " + trainingDataset['review'].size)

    # begin working with the vectorizer

    vectorizer = CountVectorizer()



def getPandasDataset(csvName, inDelimiter = "\,", header = 0):
    pandas.set_option('display.max_colWidth', 1000)
    return pandas.read_csv(csvName, header = 0, delimiter = inDelimiter, quoting = 3)

def cleanDataset(data):
    # remove the HTML
    noHTML = BS(data, "lmxl").get_text()
    # remove regular expressions
    noRegExHTML = re.sub("[^a-zA-Z]", " ", noHTML)
    # put everything to lowercase letters and split them into individual words
    lowerWords = noRegExHTML.lower().split()
    # put everything into a set for faster searching
    words = set(stopwords.lowerWords("english"))
    # remove all the stop-words from the set
    cleanedText = set()
    for word in words:
        if word not in words:
            cleanedText = word  # perfectText = [w for w in words if not w in words]

def polarityToWord(polarity):
    if 0.6 <= polarity <= 1.0:
        return 'very Positive'
    elif 0.3 <= polarity < 0.6:
        return 'positive'
    elif -0.3 <= polarity < 0.3:
        return 'undetermined'
    elif -0.6 <= polarity < -0.3:
        return 'negative'
    elif -1.0 <= polarity < 0.6:
        return 'very Negative'
    else:
        return 'na'

