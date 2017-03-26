import sys
import pandas
from bs4 import BeautifulSoup as BS
import re
from nltk.corpus import stopwords  # Import the stop word list
from sklearn.feature_extraction.text import CountVectorizer

# prepare data for learning
def main():
    trainingIterations = 100
    trainingDataset = getPandasDataset('../Data/labeledTrainData.tsv', "\t")
    cleanedTrainingData = []
    for i in range(0, trainingDataset["review"].size):
        cleanedTrainingData.append(cleanDataset(trainingDataset["review"][i]))
        if (i + 1) % 500 == 0:
            print("cleaned review %i of %i" % ((i + 1), trainingDataset['review'].size))

    # export the finished product to a .tsv file
    pandas.DataFrame(cleanedTrainingData).to_csv("../Data/labeledCleanedTrainData.tsv", sep='\t')

    # begin working with the vectorizer

    vectorizer = CountVectorizer()


def getPandasDataset(csvName, inDelimiter="\,", header=0):
    pandas.set_option('display.max_colWidth', 1000)
    return pandas.read_csv(csvName, header, delimiter=inDelimiter, quoting=3)


def cleanDataset(data):
    # remove the HTML
    noHTML = BS(data, "lxml").get_text()
    # remove regular expressions
    noRegExHTML = re.sub("[^a-zA-Z]", " ", noHTML)
    # put everything to lowercase letters and split them into individual words
    lowerWords = noRegExHTML.lower().split()
    # get all possible stopwords
    stopWords = set(stopwords.words("english"))
    # remove all the stop-words from the set
    cleanedText = [w for w in lowerWords if not w in stopWords]   # perfectText = [w for w in words if not w in words]
    return " ".join(cleanedText)

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

main()