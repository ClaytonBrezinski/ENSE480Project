import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from SentimentAnalyzer import utility
from SentimentAnalyzer import WordtoVector


def create_bag_of_centroids(wordList, wordCentroidMap):
    """
    Create a bag of centroids based off a given cluster size, and the map containing all words
    :param wordList:
    :param wordCentroidMap:
    :return:
    """
    # The number of clusters is equal to the highest cluster index in the word / centroid map
    numberOfCentroids = max(wordCentroidMap.values()) + 1

    # Pre-allocate the bag of centroids vector (for speed)
    bagOfCentroids = np.zeros(numberOfCentroids, dtype="float32")

    # Loop over the words in the review. If the word is in the vocabulary, find which cluster it belongs to, and
    # increment that cluster count by one
    for word in wordList.split():
        if word in wordCentroidMap:
            index = wordCentroidMap[word]
            bagOfCentroids[index] += 1

    # Return the "bag of centroids"
    return bagOfCentroids


def wordToVectorCentroids(trainingFilename, cleanedTrainingFileName, unlabeledTrainingFilename,
                          cleanedUnlabeledTrainingFilename, testingFilename, testingCleanedFilename,
                          premadeModelName=None):
    """
    When testing, create vectors out of each of the words within the given sentence, then through the use of clustering,
    cluster the vectors into centroids and make a sentiment guess based off of teh centroid data
    :param trainingFilename:
    :param cleanedTrainingFileName:
    :param unlabeledTrainingFilename:
    :param cleanedUnlabeledTrainingFilename:
    :param testingFilename:
    :param testingCleanedFilename:
    :param premadeModelName:
    :return:
    """

    # Read data from files
    cleanTrainData = utility.getCleanDataset(regularFilename=trainingFilename,
                                             cleanedFilename=cleanedTrainingFileName)
    cleanTestData = utility.getCleanDataset(regularFilename=testingFilename,
                                            cleanedFilename=testingCleanedFilename)

    cleanTrainData.fillna(value="NOTHING", inplace=True)
    cleanTestData.fillna(value="NOTHING", inplace=True)

    if premadeModelName is not None:
        print("Using a premade wordToVector model")
        model = Word2Vec.load(premadeModelName)
    else:
        print("creating a wordToVector model")
        model = WordtoVector.createWordToVectorModel(cleanedTrainingFileName=cleanedTrainingFileName,
                                                     trainingFilename=trainingFilename,
                                                     unlabeledTrainingFilename=unlabeledTrainingFilename,
                                                     cleanedUnlabeledTrainingFilename=cleanedUnlabeledTrainingFilename,
                                                     numberOfFeatures=300)

    """ Run k-means on the word vectors and print a few clusters """

    # Set the number of clusters to be 1/5th of the vocabulary size, or an average of 5 words per cluster
    wordVectors = model.wv.syn0
    numberOfClusters = wordVectors.shape[0] // 5

    # Initialize a k-means object and use it to extract centroids
    print("Running K means clustering ")

    kMeansClustering = KMeans(n_clusters=numberOfClusters)
    idx = kMeansClustering.fit_predict(wordVectors)

    # Create a Word / Index dictionary, mapping each vocabulary word to a cluster number
    wordCentroidMap = dict(zip(model.wv.index2word, idx))

    for cluster in range(0, 10):
        #
        # Print the cluster number
        print("\nCluster %d" % cluster)
        #
        # Find all of the words for that cluster number, and print them out
        words = []
        for word, frequency in wordCentroidMap.items():
            if frequency == cluster:
                words.append(word)
        print(words)

    """ Create bags of centroids """

    # Pre-allocate an array for the training set bags of centroids (for speed) then ransform the training set reviews
    # into bags of centroids
    trainCentroids = np.zeros((cleanTrainData["review"].size, numberOfClusters), dtype="float32")

    counter = 0
    for review in cleanTrainData["review"]:
        trainCentroids[counter] = create_bag_of_centroids(review, wordCentroidMap)
        counter += 1

    # Repeat for test reviews
    testCentroids = np.zeros((cleanTestData["review"].size, numberOfClusters), dtype="float32")
    counter = 0
    for review in cleanTestData:
        testCentroids[counter] = create_bag_of_centroids(review, wordCentroidMap)
        counter += 1

    """ Fit a random forest and extract predictions """
    forest = RandomForestClassifier(n_estimators=100)

    # Fitting the forest may take a few minutes
    print("Fitting a random forest to labeled training data...")
    forest.fit(trainCentroids, cleanTrainData["sentiment"])
    result = forest.predict(testCentroids)

    utility.printOutResults(inputData=cleanTestData, resultData=result, resultDataName="sentiment",
                            outputFilename="wordToVectorCentroids")
