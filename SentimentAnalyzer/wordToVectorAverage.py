import numpy as np  # Make sure that numpy is imported
from sklearn.ensemble import RandomForestClassifier
from SentimentAnalyzer import utility
from SentimentAnalyzer import WordtoVector
from gensim.models import Word2Vec
import scipy.stats as stats


def makeFeatureVector(words, model, numberOfFeatures):
    """
    Function to average all of the word vectors in a given paragraph
    :param words:
    :param model:
    :param numberOfFeatures:
    :return:
    """
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((numberOfFeatures,), dtype="float32")

    numberOfWords = 0.
    index2word_set = set(model.wv.index2word)

    # Loop over each word in the review and, if it is in the model's vocabulary, add its feature vector to the total
    # Then divide the result by the number of words to get the average
    for word in words:
        if word in index2word_set:
            numberOfWords += 1
            featureVec = np.add(featureVec, model[word])
    featureVec = np.divide(featureVec, numberOfWords)
    return featureVec


def getAvgFeatureVectors(reviews, model, numberOfFeatures):
    """
    When given a set of reviews, calculate the average vector for each one and return it as a 2D numpy array
    :param reviews:
    :param model:
    :param numberOfFeatures:
    :return:
    """
    counter = 0
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVectors = np.zeros((len(reviews), numberOfFeatures), dtype="float32")

    # Loop through the reviews
    for review in reviews:
        if counter % 1000 == 0.:
            print("Feature vector %i of %d complete" % (counter, len(reviews)))

        # Call the function (defined above) that makes average feature vectors
        reviewFeatureVectors[int(counter)] = makeFeatureVector(review, model, numberOfFeatures)

        counter += 1
    return reviewFeatureVectors


def getCleanReviews(reviews):
    """
    takes in the given list of reviews and returns a cleaned version of the reviews
    :param reviews:
    :return:
    """
    cleanReviewList = []
    for review in reviews:
        cleanReview = utility.cleanDataset(review, removeStopwords=True)
        cleanReviewList.append(cleanReview)
    return cleanReviewList


def wordToVectorAverage(trainingFilename, cleanedTrainingFilename, unlabeledTrainingFilename,
                        cleanedUnlabeledTrainingFilename, cleanedTestingFilename, testingFilename,
                        premadeModelName=None):
    """
    average the vec
    :param trainingFilename:
    :param testingFilename:
    :param premadeModelName:
    :return:
    """

    train = utility.getCleanDataset(trainingFilename, cleanedTrainingFilename)
    test = utility.getCleanDataset(testingFilename, cleanedTestingFilename)
    train.fillna(value="NOTHING", inplace=True)
    test.fillna(value="NOTHING", inplace=True)

    numberOfFeatures = 300

    if premadeModelName is not None:
        model = Word2Vec.load(premadeModelName)
    else:
        model = WordtoVector.createWordToVectorModel(cleanedTrainingFileName=cleanedTrainingFilename,
                                                     unlabeledTrainingFilename=unlabeledTrainingFilename,
                                                     trainingFilename=trainingFilename,
                                                     cleanedUnlabeledTrainingFilename=cleanedUnlabeledTrainingFilename,
                                                     numberOfFeatures=numberOfFeatures)

    """ Create average vectors for the training and test sets """
    print("Creating average feature vectors for training reviews")
    trainDataVectors = getAvgFeatureVectors(getCleanReviews(train["review"].tolist()), model, numberOfFeatures)

    print("Creating average feature vectors for test reviews")
    testDataVectors = getAvgFeatureVectors(getCleanReviews(test["review"].tolist()), model, numberOfFeatures)

    """ Fit a random forest to the training set, then make predictions """
    # Fit a random forest to the training data, using 100 trees
    forest = RandomForestClassifier(n_estimators=100)

    print("Fitting a random forest to labeled training data...")
    forest.fit(trainDataVectors, train["sentiment"])

    # Test & extract results
    result = forest.predict(testDataVectors)
    utility.printOutResults(inputData=test, resultData=result, resultDataName="sentiment",
                            outputFilename="wordToVectorAverage")
