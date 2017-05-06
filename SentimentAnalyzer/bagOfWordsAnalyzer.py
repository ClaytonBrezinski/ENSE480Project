import pandas
from SentimentAnalyzer import utility
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import time


def bagOfWordsAnalysis(trainingFilename, cleanedTrainingFileName, testingFilename, testingCleanedFilename):
    """
    Create a bag of words model on the training data provided and then uses said model to classify the testing data
    provided
    :param trainingFilename: name of the file we will train the bag of words from
    :param cleanedTrainingFileName: name of the potentially cleaned training file
    :param testingFilename: name of the file we will test bag of words from
    :param testingCleanedFilename: name of the potentially cleaned testing file
    :return:
    """
    # get the training data and clean it
    cleanedTrainingData = utility.getCleanDataset(trainingFilename, cleanedTrainingFileName)

    # Setup the Vectorizer to a word analyzer with a maximum of 5000 features and no stopwords being accounted for
    vectorizer = CountVectorizer(stop_words=None, analyzer="word", tokenizer=None, preprocessor=None,
                                 max_features=5000)
    test = cleanedTrainingData["review"].tolist()
    # fit the model then learn the vocabulary
    print("Training model to the training Data")
    train_data_features = vectorizer.fit_transform(test)

    # Create a random forest classifier with 100 trees.
    # Fit the forest to the training set, using the bag of words as features and the sentiment numbers as values
    print("Building random forest classifier...")
    timeStart = time.clock()
    forest = RandomForestClassifier(n_estimators=100, n_jobs=-1)  # n_job = number of cores available, -1 is use all
    forest.fit(train_data_features, cleanedTrainingData['sentiment'])
    timeEnd = time.clock()
    print("Random forest classifier built with %i seconds, taking data now" % (timeStart - timeEnd))

    # Read the test data
    cleanedTestData = utility.getCleanDataset(testingFilename, testingCleanedFilename)

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(cleanedTestData["review"])
    test_data_features = test_data_features.toarray()

    # Use the random forest to make sentiment label predictions
    result = forest.predict(test_data_features)

    # Copy the results to a pandas dataframe with an "id" column and a "sentiment" column
    output = pandas.DataFrame(
        data={"id": cleanedTestData["id"],
              "sentiment": result,
              "review": cleanedTestData["review"]})

    # Use pandas to write the comma-separated output file
    print("Bag of Words Classifier output complete")
    output.to_csv("BagOfWordsClassified.csv", index=False, quoting=3, sep='\t')
