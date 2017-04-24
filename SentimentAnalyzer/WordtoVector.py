import nltk.data
import logging
from SentimentAnalyzer import utility
from gensim.models import Word2Vec


def createWordToVectorModel(cleanedTrainingFileName, unlabeledTrainingFilename, trainingFilename,
                            cleanedUnlabeledTrainingFilename, numberOfFeatures=300,
                            resultingModelName="WordToVectorModel"):
    """
    this method creates a word to vector model based upon the labeled and unlabeled training data that is presented to it.
    :param numberOfFeatures: the amount of dimensions that we want our word to vector system to build to
    :param resultingModelName: the name of the file that the model will sit in
    :param trainingFilename: the name of the file used for training
    :param cleanedTrainingFileName: the potential name of a training file that has already been cleaned
    :param unlabeledTrainingFilename: the name of the file used for unlabeled training
    :param cleanedUnlabeledTrainingFilename: the potential name of a unlabeled training file that has already been cleaned
    :return:
    """
    # Read data from files
    trainingData = utility.getCleanDataset(cleanedTrainingFileName, trainingFilename)
    unlabeledTrainingData = utility.getCleanDataset(cleanedUnlabeledTrainingFilename, unlabeledTrainingFilename)

    # Load the punkt tokenizer from the english language pickle file.
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    """ Split the labeled and unlabeled training sets into clean sentences """
    sentences = []

    print("Parsing sentences from training set")
    i = 0
    for review in trainingData["review"]:
        i += 1
        if i % 1000 == 0:
            print("% i of %i sentences complete" % (i, len(trainingData)))
            sentences += utility.paragraphToSentences(data=review, tokenizer=tokenizer)

    print("Parsing sentences from unlabeled set")
    i = 0
    for review in unlabeledTrainingData["review"]:
        i += 1
        if i % 1000 == 0:
            print("% i of %i sentences complete" % (i, len(trainingData)))
            sentences += utility.paragraphToSentences(data=review, tokenizer=tokenizer)

    """ Set parameters and train the word2vec model """
    # Import the built-in logging module and configure it so that Word2Vec creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Set values for various parameters
    numberOfFeatures = numberOfFeatures  # Word vector dimensionality
    minimumWordCount = 40  # Minimum word count for the model
    numberOfWorkers = 4  # how many processes we want to run
    context = 10  # Context window size
    downSampling = 1e-3  # the setting for down sampling frequently occurring word

    # Initialize and train the model (this will take some time)
    print("Training Word2Vec model...")
    model = Word2Vec(sentences, workers=numberOfWorkers, size=numberOfFeatures, min_count=minimumWordCount,
                     window=context,
                     sample=downSampling, seed=1)

    # Since I don't want to train the model any more, lock in everything so that we are process efficient.
    model.init_sims(replace=True)

    # Save the model for future, faster use by saving it and then using Word2Vec.load(<FILENAME>) to pick it up later
    modelName = resultingModelName
    model.save(modelName)
    return model
