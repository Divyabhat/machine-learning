from collections import defaultdict
import math
import random
import numpy
import sys

class NaiveBayes:
    def __init__(self, trainingfile, testfile):
        self.valueMatrix = defaultdict(lambda: defaultdict(list))
        self.meanMatrix = defaultdict(lambda: defaultdict(float))
        self.sdMatrix = defaultdict(lambda: defaultdict(float))

        self.classCount = {}
        self.classProb = {}
        self.totalClass = 0

        self.trainingFilePath = trainingfile
        self.testFilePath = testfile

    def readTrainingFileAndProcess(self):
        fileContent = open(self.trainingFilePath, 'r')

        for line in fileContent:
            self.processLine(line)

    def processLine(self, line):
        nums = [float(n) for n in line.split()]

        classNum = int(nums[-1])
        self.totalClass += 1

        if classNum in self.classCount:
            self.classCount[classNum] += 1
        else:
            self.classCount[classNum] = 1

        for i in range(0, len(nums) - 1):
            self.valueMatrix[classNum][i].append(nums[i])

    def calculateMeanAndSD(self):
        for classIndex in self.classCount:
            for featureIndex in range(len(self.valueMatrix[classIndex])):
                self.meanMatrix[classIndex][featureIndex] = numpy.mean(self.valueMatrix[classIndex][featureIndex])
                sdTemp = numpy.std(self.valueMatrix[classIndex][featureIndex])
                self.sdMatrix[classIndex][featureIndex] = sdTemp
                if sdTemp < 0.01:
                    self.sdMatrix[classIndex][featureIndex] = 0.01

    def  printTrainingData(self):
        for classIndex in self.classCount:
            for featureIndex in range(len(self.valueMatrix[classIndex])):
                print("Class %3d, attribute %d, mean = %.5f, std = %.5f" % (classIndex, featureIndex+1, self.meanMatrix[classIndex][featureIndex], self.sdMatrix[classIndex][featureIndex]))

    def calculateClassProb(self):
        for classIndex in self.classCount:
            self.classProb[classIndex] = self.classCount[classIndex] / self.totalClass

    def processTrainingSet(self):
        nb.readTrainingFileAndProcess()
        nb.calculateMeanAndSD()
        nb.calculateClassProb()
        nb.printTrainingData()

    def calculateProbability(self, xi, mean, sd):
        if sd == 0:
            return 0

        expVar = -1 * (math.pow((xi - mean), 2)) / (2 * sd * sd)
        expValue = math.exp(expVar)

        constValue = 1.0 / (math.sqrt(2 * numpy.pi) * sd)
        prob = constValue * expValue
        return prob

    def processTestSet(self):
        fileContent = open(self.testFilePath, 'r')

        linenumber = 1
        classificationAccuracy = 0.0
        for line in fileContent:
            nums = [float(n) for n in line.split()]

            actualClass = int(nums[-1])

            maxProb = -100.0
            maxProbList = []
            for classIndex in self.classCount:
                prob = self.classProb[classIndex]
                for featureIndex in range(len(nums) - 1):
                    prob = prob * self.calculateProbability(nums[featureIndex], self.meanMatrix[classIndex][featureIndex], self.sdMatrix[classIndex][featureIndex])

                if prob > maxProb:
                    maxProb = prob
                    maxProbList = []
                    maxProbList.append(classIndex)
                elif prob == maxProb:
                    maxProbList.append(classIndex)

            predClass = random.choice(maxProbList)

            accuracy = 0.0
            if len(maxProbList) == 1:
                if actualClass == predClass:
                    accuracy = 1.0
                else:
                    accuracy = 0.0
            elif len(maxProbList) > 1:
                accuracy = 1.0 / float(len(maxProbList))

            print("ID=%3d, predicted=%2d, probability=%13.4f, true=%d, accuracy=%4.2f" % (linenumber, predClass, maxProb, actualClass, accuracy))

            classificationAccuracy += accuracy
            linenumber += 1
        classificationAccuracy = classificationAccuracy / linenumber
        print("Classification accuracy = %6.4f" % (classificationAccuracy * 100))

if __name__ == "__main__":
    nb = NaiveBayes("./yeast_training.txt", "./yeast_test.txt")
    #nb = NaiveBayes("./satellite_training.txt", "./satellite_test.txt")
    #nb = NaiveBayes("./pendigits_training.txt", "./pendigits_test.txt")

    nb.processTrainingSet()
    nb.processTestSet()


