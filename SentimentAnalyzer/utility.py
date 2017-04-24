import os.path
import re
import pandas
from bs4 import BeautifulSoup as BS
from nltk.corpus import stopwords


def getPandasDataset(csvName, inDelimiter="\,", header=0):
    """
    Take in the csv name, get the pandas dataset depending on what the delimiter and header value is
    :param csvName:
    :param inDelimiter:
    :param header:
    :return:
    """
    pandas.set_option('display.max_colWidth', 1000)
    return pandas.read_csv(csvName, header, delimiter=inDelimiter, quoting=3)


def printOutResults(inputData, resultData, resultDataName, outputFilename):
    """
    print out the results of the given data to a .CSV file
    :param inputData:
    :param resultData:
    :param resultDataName:
    :param outputFilename:
    :return:
    """
    # Write the test results
    output = pandas.DataFrame(data={"id": inputData["id"], resultDataName: resultData, "review": inputData["review"]})
    output.to_csv(str(outputFilename) + ".csv", index=False, quoting=3, sep='\t')
    print("Wrote " + str(outputFilename) + ".csv")
    return


def polarityToWordZeroToFour(polarity) -> str:
    """
    change the input polarity to words
    :param polarity:
    :return: string
    """
    if polarity == 0:
        return 'negative'
    elif polarity == 1:
        return 'somewhat negative'
    elif polarity == 2:
        return 'neutral'
    elif polarity == 3:
        return 'somewhat positive'
    elif polarity == 4:
        return 'positive'
    else:
        return 'na'


def polarityToWordNegativeOneToOne(polarity) -> str:
    """
    change the input polarity to words
    :param polarity:
    :return: string
    """
    if 0.6 <= polarity <= 1.0:
        return 'very Positive'
    elif 0.3 <= polarity < 0.6:
        return 'positive'
    elif -0.3 <= polarity < 0.3:
        return 'neutral'
    elif -0.6 <= polarity < -0.3:
        return 'negative'
    elif -1.0 <= polarity < 0.6:
        return 'very Negative'
    else:
        return 'na'


def cleanDataset(data, removeStopwords=False):
    """
    Remove the html of the input data and optionally remove all the stopwords as well
    :param data:
    :param removeStopwords:
    :return:
    """
    # remove the HTML
    noHTML = BS(data, "lxml").get_text()
    # remove regular expressions
    noRegExHTML = re.sub("[^a-zA-Z]", " ", noHTML)
    # put everything to lowercase letters and split them into individual words
    cleanedText = noRegExHTML.lower().split()
    # get all possible stopwords
    if removeStopwords:
        stopWords = set(stopwords.words("english"))
        # remove all the stop-words from the set
        cleanedText = [w for w in cleanedText if not w in stopWords]
    return " ".join(cleanedText)


def paragraphToSentences(data, tokenizer, removeStopwords=False):
    """
    Break down the input paragraphs into smaller sentences and then clean the dataset
    :param data:
    :param tokenizer:
    :param removeStopwords:
    :return:
    """
    # break down a large paragram into smaller sentences
    rawSentences = tokenizer.tokenize(data.strip())
    sentences = []
    for rawSentence in rawSentences:
        if len(rawSentence) > 0 and rawSentence != "" and str(rawSentence) != 'nan':
            # clean up all the non-empty sentences
            sentences.append(cleanDataset(rawSentence, removeStopwords))
    # Return the list of sentences
    # (each sentence is a list of words, so this returns a list of lists)
    return sentences


def getCleanDataset(regularFilename, cleanedFilename) -> pandas.DataFrame:
    """

    :param regularFilename:
    :param cleanedFilename:
    :return: DataFrame containing columns: id, sentiment, cleaned review files
    """
    if os.path.exists("../Data/" + str(cleanedFilename)):
        print("pre-cleaned data found!")
        cleanedDataFrame = getPandasDataset("../Data/" + str(cleanedFilename), "\t")
    elif os.path.exists("../Data/" + str(regularFilename)):
        print("pre-cleaned data file named %s not found. Cleaning the data, placing it into an array, and writing a cleaned file!" %cleanedFilename)
        trainingDataset = getPandasDataset('../Data/' + str(regularFilename), "\t")
        cleanedReview = []
        for review in trainingDataset["review"]:
            cleanedReview.append(cleanDataset(review, removeStopwords=True))
            if (len(cleanedReview) + 1) % 1000 == 0:
                print("cleaned review %i of %i" % ((len(cleanedReview) + 1), trainingDataset['review'].size))

        cleanedDataFrame = pandas.DataFrame({"id": trainingDataset['id'].tolist(),
                                     "review": cleanedReview})
        if "sentiment" in trainingDataset.columns:
            cleanedDataFrame["sentiment"] = trainingDataset["sentiment"].tolist()

        # export the finished product to a .tsv file
        dataFrame = pandas.DataFrame(data=cleanedDataFrame)
        pandas.DataFrame(dataFrame).to_csv("../Data/" + str(cleanedFilename), sep='\t', index=False)
    else:
        print("Required files not found %s, %s" % regularFilename % cleanedFilename)
        exit(0)
    return cleanedDataFrame
