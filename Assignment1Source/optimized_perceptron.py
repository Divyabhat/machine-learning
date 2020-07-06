import csv
import numpy as np
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import datetime


class DigitIdentifier:
    def __init__(self, learningRate):
        self.trainingDataFile = "./mnist_train.csv"
        self.trainingArray = np.array([])
        self.trainingAccuracy = []

        self.validationDataFile = "./mnist_validation.csv"
        self.validationArray = np.array([])
        self.validationAccuracy = []

        self.weights = np.random.uniform(-0.05, 0.05, (10, 785))
        self.learningRate = learningRate

        self.actualList = []
        self.predList = []

        self.trainingTargetClassList = []
        self.validationTargetClassList = []

    def load_csv(self, filepath):
        fd = open(filepath, 'r')
        fileContent = csv.reader(fd)
        fileContentListStr = list(fileContent)
        fileContentListInt = [list(map(int,i) ) for i in fileContentListStr]
        storedArray = np.array(fileContentListInt)

        normalizedArray = np.divide(storedArray, float(255))

        targetClassList = []
        for i in range(storedArray.shape[0]):
            targetClassList.append(storedArray[i, 0].astype('int'))
            normalizedArray[i, 0] = 1

        return normalizedArray, targetClassList

    def load_training_and_validation_file(self):
        self.trainingArray, self.trainingTargetClassList = self.load_csv(self.trainingDataFile)
        self.validationArray, self.validationTargetClassList = self.load_csv(self.validationDataFile)

    def fill_target_list(self, targetClass):
        targetList = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        targetList[targetClass] = 1
        return targetList

    def calculate_yvalue(self, percepValue):
        if (percepValue <= 0):
            return 0
        else:
            return 1

    def learn_and_find_accuracy(self, epoch, dataArray, targetClassList, trainingExecution):
        self.predList = []
        self.actualList = []

        for i in range(0, dataArray.shape[0]):
            targetClass = targetClassList[i]
            targetList = self.fill_target_list(targetClass)
            #print("Target class = ", targetClass)

            percepList = []
            yList = []
            self.actualList.append(targetClass)

            for perceptronIndex in range(10):
                percepValue = np.dot(dataArray[i], self.weights[perceptronIndex, :])
                yValue = self.calculate_yvalue(percepValue)
                percepList.append(percepValue)
                yList.append(yValue)

                if trainingExecution and epoch > 0:
                    deltaWeight = self.learningRate * (targetList[perceptronIndex] - yList[perceptronIndex]) * dataArray[i]
                    self.weights[perceptronIndex, :] = self.weights[perceptronIndex, :] + deltaWeight

            self.predList.append(np.argmax(np.array(percepList)))

        accuracy = ((np.array(self.predList) == np.array(self.actualList)).sum() / float(len(self.actualList))) * 100

        return accuracy

    def store_accuracy(self, filepath, accuracyIndex, accuracy):
        with open(filepath, 'a', newline="") as fd:
            csvWriter = csv.writer(fd)
            csvWriter.writerow([accuracyIndex, accuracy])

    def execute_trained_data(self, epoch):
        accuracy = self.learn_and_find_accuracy(epoch, self.trainingArray, self.trainingTargetClassList, True)
        self.trainingAccuracy.append(accuracy)
        self.store_accuracy('train_output' + str(self.learningRate) + '.csv', epoch, accuracy)

    def execute_validation_data(self, epoch):
        accuracy = self.learn_and_find_accuracy(epoch, self.validationArray, self.validationTargetClassList ,False)
        self.validationAccuracy.append(accuracy)
        self.store_accuracy('validation_output' + str(self.learningRate) + '.csv', epoch, accuracy)
        print("Learning rate = %s, Validation Accuracy for epoch %s = %s" % (self.learningRate, epoch, accuracy))

    def learn_digits(self):
        for epoch in range(50):
            self.execute_trained_data(epoch)
            self.execute_validation_data(epoch)
        print("Confusion Matrix: ")
        print(confusion_matrix(self.actualList, self.predList))

    def plot_graph(self):
        trainFile = 'train_output' + str(self.learningRate) + '.csv'
        validationFile = 'validation_output' + str(self.learningRate) + '.csv'

        trainX, trainY = np.loadtxt(trainFile, delimiter=',', unpack=True)
        validationX, validationY = np.loadtxt(validationFile, delimiter=',', unpack=True)

        plt.plot(trainX, trainY, label="Training Set")
        plt.plot(validationX, validationY, label="Testing Set")

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%) ')
        plt.title('For Learning rate ' + str(self.learningRate))
        plt.legend()
        plt.show()

if __name__ == "__main__":
    for learningRate in (0.1, 0.01, 0.001):
        print("Starting time for learning rate %s = %s" % (learningRate, datetime.datetime.now().time()))
        digitIdentifierObj = DigitIdentifier(learningRate)
        digitIdentifierObj.load_training_and_validation_file()
        digitIdentifierObj.learn_digits()
        digitIdentifierObj.plot_graph()
        print("Ending time for learning rate %s = %s" % (learningRate, datetime.datetime.now().time()))