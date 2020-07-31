from collections import defaultdict
import math
import random
import statistics

class NaiveBayes:
    def __init__(self):
        self.valueMatrix = defaultdict(lambda: defaultdict(list))
        self.meanMatrix = defaultdict(lambda: defaultdict())
        self.sdMatrix = defaultdict(lambda: defaultdict())

        self.classCount = {}
        self.classProb = {}
        self.totalClass = 0

        self.trainingFilePath = "./yeast_training.txt"
        self.testFilePath = "./yeast_test.txt"

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
        for classIndex in range(1, len(self.classCount)+1):
            for featureIndex in range(len(self.valueMatrix[classIndex])):
                self.meanMatrix[classIndex][featureIndex] = statistics.mean(self.valueMatrix[classIndex][featureIndex])
                sdTemp = statistics.stdev(self.valueMatrix[classIndex][featureIndex])
                self.sdMatrix[classIndex][featureIndex] = sdTemp
                if sdTemp < 0.01:
                    self.sdMatrix[classIndex][featureIndex] = 0.01

    def  printTrainingData(self):
        for classIndex in range(1, len(self.classCount)+1):
            for featureIndex in range(len(self.valueMatrix[classIndex])):
                print("Class %d, attribute %d, mean = %.2f, std = %.2f" % (classIndex, featureIndex+1, self.meanMatrix[classIndex][featureIndex], self.sdMatrix[classIndex][featureIndex]))

        print(self.totalClass)
        print(self.classCount)
        print(self.classProb)

    def calculateClassProb(self):
        for i in range(1, len(self.classCount)+1):
            self.classProb[i] = self.classCount[i] / self.totalClass

    def processTrainingSet(self):
        nb.readTrainingFileAndProcess()
        nb.calculateMeanAndSD()
        nb.calculateClassProb()
        nb.printTrainingData()

    def calculateProbability(self, xi, mean, sd):
        exponent = math.exp(-(math.pow(xi - mean, 2) / (2 * math.pow(sd, 2))))
        return (1 / (math.sqrt(2 * math.pi) * sd)) * exponent

    def processTestSet(self):
        fileContent = open(self.testFilePath, 'r')

        linenumber = 1
        classificationAccuracy = 0.0
        for line in fileContent:
            nums = [float(n) for n in line.split()]

            actualClass = int(nums[-1])

            probList = []
            for classIndex in range(1, len(self.classCount)+1):
                prob = self.classProb[classIndex]
                for featureIndex in range(len(nums) - 1):
                    prob = prob * self.calculateProbability(nums[featureIndex], self.meanMatrix[classIndex][featureIndex], self.sdMatrix[classIndex][featureIndex])
                probList.append(prob)

            maxProb = max(probList)
            maxIndexList =  [i for i, j in enumerate(probList) if j == maxProb]

            selectedIndex = random.choice(maxIndexList)

            accuracy = 0.0
            if len(maxIndexList) == 1:
                if (selectedIndex + 1) == actualClass:
                    accuracy = 1.0
                else:
                    accuracy = 0.0
            else:
                accuracy = 1 / len(maxIndexList)


            classificationAccuracy += accuracy

            print("ID=%5d, predicted=%3d, probability=%.4f, true=%3d, accuracy=%4.2f" % (linenumber, selectedIndex+1, probList[selectedIndex], actualClass, accuracy))

            linenumber += 1
        classificationAccuracy = classificationAccuracy / linenumber
        print("Classification accuracy = %6.4f" % (classificationAccuracy))

if __name__ == "__main__":
    nb = NaiveBayes()
    nb.processTrainingSet()
    nb.processTestSet()