from SentimentAnalyzer import bagOfWordsAnalyzer
from SentimentAnalyzer import wordToVectorAverage
from SentimentAnalyzer import wordToVectorCentroids


# prepare data for learning
def main():

    print("Welcome to the sentiment classifier, which model would you like to use? ")
    print("A: bag of words")
    print("B: word to vector average")
    print("C: word to vector centroid")
    #choice = input()
    choice = 'c'
    while choice != "A" and choice != "B" and choice != "C" and choice != "a" and choice != "b" and choice != "c":
        print("please select a valid choice")
        #choice = input()
    if choice == "A" or choice == "a":
        bagOfWordsAnalyzer.bagOfWordsAnalysis(trainingFilename="train.tsv", cleanedTrainingFileName="cleanedTrain.tsv",
                                              testingFilename="test.tsv", testingCleanedFilename="cleanedTest.tsv")
    elif choice == "B" or choice == "C" or choice == "b" or choice == "c":
        print("would you like to use a premade model? Y: yes, N: no")
        #choice2 = input()
        choice2 = 'n'
        while choice2 != "Y" and choice2 != "N" and choice2 != "y" and choice2 != "n":
            print("please select a valid choice")
            choice2 = input()
        if choice2 == "Y" or choice2 == "y":
            if choice == "B" or choice == "b":
                wordToVectorAverage.wordToVectorAverage(trainingFilename="train.tsv",
                                                        cleanedTrainingFilename="cleanedTrain.tsv",
                                                        cleanedTestingFilename="cleanedTest.tsv",
                                                        testingFilename="test.tsv",
                                                        unlabeledTrainingFilename="unlabeledTrainData.tsv",
                                                        cleanedUnlabeledTrainingFilename="cleanedUnlabeledTrainData.tsv",
                                                        premadeModelName="WordToVectorModel")
            else:
                wordToVectorCentroids.wordToVectorCentroids(trainingFilename="train.tsv",
                                                            cleanedTrainingFileName="cleanedTrain.tsv",
                                                            testingFilename="test.tsv",
                                                            testingCleanedFilename="cleanedTest.tsv",
                                                            premadeModelName="WordToVectorModel")
        else:
            if choice == "B" or "b":
                wordToVectorAverage.wordToVectorAverage(trainingFilename="train.tsv",
                                                        cleanedTrainingFilename="cleanedTrain.tsv",
                                                        cleanedTestingFilename="cleanedTest.tsv",
                                                        unlabeledTrainingFilename="unlabeledTrainData.tsv",
                                                        cleanedUnlabeledTrainingFilename="cleanedUnlabeledTrainData.tsv",
                                                        testingFilename="test.tsv")
            else:
                wordToVectorCentroids.wordToVectorCentroids(trainingFilename="train.tsv",
                                                            cleanedTrainingFileName="cleanedTrain.tsv",
                                                            testingFilename="test.tsv",
                                                            testingCleanedFilename="cleanedTest.tsv")


main()
