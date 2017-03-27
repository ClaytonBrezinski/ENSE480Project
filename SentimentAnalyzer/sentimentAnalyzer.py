import os.path
import sys
import pandas
import numpy as np
from bs4 import BeautifulSoup as BS
import re
from nltk.corpus import stopwords  # Import the stop word list
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


# prepare data for learning
def main():
    bagOfWordsAnalysis()


def wordToVectorAnalysis():
    pass

def bagOfWordsAnalysis():
    if os.path.exists("../Data/labeledCleanedTrainData.tsv"):
        trainingDataset = getPandasDataset('../Data/labeledTrainData.tsv', "\t")
        cleanedTrainingData = getPandasDataset("../Data/labeledCleanedTrainData.tsv")
        data = []
        for i in range(0, trainingDataset["review"].size):
            data.append(cleanedTrainingData["review"][i])
            if (i + 1) % 500 == 0:
                print("cleaned review %i of %i" % ((i + 1), trainingDataset['review'].size))
        cleanedTrainingData = data
    else:
        trainingDataset = getPandasDataset('../Data/labeledTrainData.tsv', "\t")
        cleanedTrainingData = []
        for i in range(0, trainingDataset["review"].size):
            cleanedTrainingData.append(cleanDataset(trainingDataset["review"][i]))
            if (i + 1) % 500 == 0:
                print("cleaned review %i of %i" % ((i + 1), trainingDataset['review'].size))
        # export the finished product to a .tsv file
        dataFrame = pandas.DataFrame(cleanedTrainingData)
        dataFrame.columns = ["review"]
        pandas.DataFrame(dataFrame).to_csv("../Data/labeledCleanedTrainData.tsv", sep='\t')

    # setup vectorizer
    vectorizer = CountVectorizer(stop_words=None, analyzer="word", tokenizer=None, preprocessor=None,
                                 max_features=5000)
    # fit the model then learn the vocabulary - then transform the training data into vectors
    train_data_features = vectorizer.fit_transform(cleanedTrainingData)

    # create a random forest classifier with 100 trees.
    # Fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable

    print("Building random forest classifier...")
    forest = RandomForestClassifier(n_estimators = 100, n_jobs=2)
    forest.fit(train_data_features, trainingDataset['sentiment'])
    print("Random forest classifier built, taking data now")

    # TEST AREA

    # Read the test data
    if os.path.exists("../Data/cleanedTestData.tsv"):
        testDataset = getPandasDataset('../Data/testData.tsv', "\t")
        cleanedTestData = getPandasDataset("../Data/cleanedTestData.tsv")
        data = []
        for i in range(0, testDataset["review"].size):
            data.append(cleanedTestData["review"][i])
            if (i + 1) % 500 == 0:
                print("cleaned review %i of %i" % ((i + 1), testDataset['review'].size))
        cleanedTestData = data
    else:
        trainingDataset = getPandasDataset('../Data/testData.tsv', "\t")
        cleanedTestData = []
        for i in range(0, trainingDataset["review"].size):
            cleanedTestData.append(cleanDataset(trainingDataset["review"][i]))
            if (i + 1) % 500 == 0:
                print("cleaned review %i of %i" % ((i + 1), trainingDataset['review'].size))
        # export the finished product to a .tsv file
        dataFrame = pandas.DataFrame(cleanedTestData)
        dataFrame.columns = ["review"]
        pandas.DataFrame(dataFrame).to_csv("../Data/cleanedTestData.tsv", sep='\t')

    # # Create an empty list and append the clean reviews one by one
    # clean_test_reviews = []
    # print ("Cleaning and parsing the test set movie reviews...\n")
    # for i in range(0, len(test["review"])):
    #     clean_review = cleanDataset(test["review"][i])
    #     if (i + 1) % 500 == 0:
    #         print ("Review %d of %d\n" % (i + 1, len(test["review"])))
    #     clean_test_reviews.append(clean_review)

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(cleanedTestData)
    test_data_features = test_data_features.toarray()

    # Use the random forest to make sentiment label predictions
    result = forest.predict(test_data_features)

    # Copy the results to a pandas dataframe with an "id" column and
    # a "sentiment" column
    output = pandas.DataFrame(data={"id": testDataset["id"], "sentiment": result})

    # Use pandas to write the comma-separated output file
    output.to_csv("Bag_of_Words_model.csv", index=False, quoting=3,  sep='\t')


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