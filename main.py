import numpy as np
import pandas as pd
import math


def readTrainFeatures():
    return pd.read_csv('./dataset/question-4-train-features.csv', header=None)


def readTrainLabels():
    return pd.read_csv('./dataset/question-4-train-labels.csv', header=None)


def readTestFeatures():
    return pd.read_csv('./dataset/question-4-test-features.csv', header=None)


def readTestLabels():
    return pd.read_csv('./dataset/question-4-test-labels.csv', header=None)


def calcEachWordOccurrence():
    trainFeatures = readTrainFeatures()
    trainLabels = readTrainLabels()

    wordCountForSpace = []
    wordCountForMedical = []

    for i in range(0, len(trainFeatures.columns)):
        wordCountForSpace.append(0)
        wordCountForMedical.append(0)

    for rowIndex, row in trainFeatures.iterrows():
        for column in range(0, len(trainFeatures.columns)):
            if trainLabels.iat[rowIndex, 0] == 1:
                wordCountForSpace[column] += trainFeatures.iat[rowIndex, column]
            if trainLabels.iat[rowIndex, 0] == 0:
                wordCountForMedical[column] += trainFeatures.iat[rowIndex, column]

    with open('./dataset/occur_space.txt', 'w') as file:
        file.writelines("%s\n" % spc for spc in wordCountForSpace)

    with open('./dataset/occur_medical.txt', 'w') as file:
        file.writelines("%s\n" % med for med in wordCountForMedical)


def readTxt(filename):
    res = []
    with open(filename, 'r') as file:
        fileContents = file.readlines()
        for line in fileContents:
            # remove linebreak which is the last character of the string
            cur = float(line[:-1])
            res.append(cur)
    return res


def calcEstimators():
    spaceWords = readTxt('./dataset/occur_space.txt')
    medicalWords = readTxt('./dataset/occur_medical.txt')

    probabilitiesSpace = []
    probabilitiesMedical = []

    for i in range(0, len(spaceWords)):
        probabilitiesSpace.append((spaceWords[i] + 1) / (np.sum(spaceWords) + len(spaceWords)))

    for i in range(0, len(medicalWords)):
        probabilitiesMedical.append((medicalWords[i] + 1) / (np.sum(medicalWords) + len(medicalWords)))

    with open('./dataset/prob_space.txt', 'w') as file:
        file.writelines("%s\n" % prob for prob in probabilitiesSpace)

    with open('./dataset/prob_medical.txt', 'w') as file:
        file.writelines("%s\n" % prob for prob in probabilitiesMedical)


# Calculate x * log(y) in special cases
def log(x, y):
    if (x == 0) and (y == 0):
        return 0
    if (x != 0) and (y == 0):
        return float('-inf')
    if (x != 0) and (y != 0):
        return x * math.log(y)
    if (x == 0) and (y != 0):
        return 0


def predict():
    trainLabels = readTrainLabels()
    trainLabelsArr = trainLabels.values

    testFeatures = readTestFeatures()
    testLabels = readTestLabels()
    testLabelsArr = testLabels.values

    totalEmails = trainLabels.size
    numOfClasses, indices = np.unique(trainLabelsArr, return_counts=True)

    n_0 = indices[0]
    n_1 = indices[1]

    pi_0 = n_0 / totalEmails
    pi_1 = n_1 / totalEmails

    theta_0 = readTxt('./dataset/prob_medical.txt')
    theta_1 = readTxt('./dataset/prob_space.txt')

    medicalProbs = []
    spaceProbs = []

    predictions = []

    # Calculate medical probability for each doc
    for i, row in testFeatures.iterrows():
        innerSum = 0
        currentDoc = testFeatures.iloc[i]

        for col in range(0, len(currentDoc)):
            innerSum += log(currentDoc[col], theta_0[col])
        innerSum += math.log(pi_0)
        medicalProbs.append(innerSum)

    # Calculate space probability for each doc
    for i, row in testFeatures.iterrows():
        innerSum = 0
        currentDoc = testFeatures.iloc[i]

        for col in range(0, len(currentDoc)):
            innerSum += log(currentDoc[col], theta_1[col])
        innerSum += math.log(pi_1)
        spaceProbs.append(innerSum)

    with open('./dataset/prob_medical_v2.txt', 'w') as file:
        file.writelines("%s\n" % prob for prob in medicalProbs)

    with open('./dataset/prob_space_v2.txt', 'w') as file:
        file.writelines("%s\n" % prob for prob in spaceProbs)

    # Compare probabilities for each doc
    for i in range(0, len(spaceProbs)):
        if spaceProbs[i] > medicalProbs[i]:
            predictions.append(1)
        if spaceProbs[i] <= medicalProbs[i]:
            predictions.append(0)

    true = 0
    for i in range(0, len(spaceProbs)):
        if predictions[i] == testLabelsArr[i]:
            true += 1
    accuracy = true / testLabelsArr.size
    print("Accuracy: %{}".format(accuracy * 100))


def main():
    calcEachWordOccurrence()
    calcEstimators()
    predict()


main()
