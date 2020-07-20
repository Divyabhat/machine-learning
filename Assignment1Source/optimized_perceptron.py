import csv
import numpy as np
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import datetime
import time

trainingDataFile = "./mnist_train.csv"
validationDataFile = "./mnist_validation.csv"

##plotting the graph for training and validation accuracy over epochs
def plot_graph(learningRate):
    trainFile = 'train_output' + str(learningRate) + '.csv'
    validationFile = 'validation_output' + str(learningRate) + '.csv'

    trainX, trainY = np.loadtxt(trainFile, delimiter=',', unpack=True)
    validationX, validationY = np.loadtxt(validationFile, delimiter=',', unpack=True)

    plt.plot(trainX, trainY, label="Training Set")
    plt.plot(validationX, validationY, label="Validation Set")

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%) ')
    plt.title('For Learning rate ' + str(learningRate))
    plt.legend()
    plt.savefig('graph' + str(learningRate) + '.png')
    plt.cla()

class FileLoader:
    def __init__(self, filepath):
        self.dataArray = []
        self.targetClassList = []
        self.filepath = filepath
        self.bias = 1

    ## Load the train and validation csv files into train data and validation data as numpy array
    ## and normalizing the data by deviding it by 255 to a fraction between 0 and 1
    def load_csv_and_normalize(self):
        fd = open(self.filepath, 'r')
        fileContent = csv.reader(fd)
        fileContentListStr = list(fileContent)
        fileContentListInt = [list(map(int,i) ) for i in fileContentListStr]
        storedArray = np.array(fileContentListInt)

        self.dataArray = np.divide(storedArray, float(255))

        self.targetClassList = []
        for i in range(storedArray.shape[0]):
            self.targetClassList.append(storedArray[i, 0].astype('int'))
            self.dataArray[i, 0] = self.bias

class DigitIdentifier:
    def __init__(self, learningRate):
        self.trainingArray = np.array([])
        self.trainingAccuracy = []

        self.validationArray = np.array([])
        self.validationAccuracy = []

        ### Initialize the weight variable of the dimension 10 x 785  where
        ### each single row of 1 x 785 is input to one single perceptron for
        ### a given training example
        self.weights = np.random.uniform(-0.05, 0.05, (10, 785))
        self.learningRate = learningRate

        self.actualList = []
        self.predList = []

        self.trainingTargetClassList = []
        self.validationTargetClassList = []

    def initialize_training_and_validation_file(self, trainingDataLoader, validationDataLoader):
        self.trainingArray = trainingDataLoader.dataArray
        self.trainingTargetClassList = trainingDataLoader.targetClassList

        self.validationArray = validationDataLoader.dataArray
        self.validationTargetClassList = validationDataLoader.targetClassList

    def fill_target_list(self, targetClass):
        targetList = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        targetList[targetClass] = 1
        return targetList

    def calculate_yvalue(self, percepValue):
        if (percepValue <= 0):
            return 0
        else:
            return 1

    ### perceptron_learn function : This function is called to train the perceptrons
    ### and weight updation purposes. The first for loop iterates through each training
    ### example and Vectorized function is used for the weight updation for all perceptrons.
    ### Parameters :
    def learn_and_find_accuracy(self, epoch, dataArray, targetClassList, trainingExecution):
        self.predList = []
        self.actualList = []

        for i in range(0, dataArray.shape[0]):
            targetClass = targetClassList[i]
            targetList = self.fill_target_list(targetClass)
            #print("Target class = ", targetClass)

            percepList = []
            self.actualList.append(targetClass)

            percepValueList = np.sum(dataArray[i] * self.weights, axis=1)
            yList = np.array(np.greater(percepValueList, 0) * 1)

            if trainingExecution and epoch > 0:
                diffList = np.subtract(targetList, yList)
                diffList = diffList.reshape(10, 1)
                #print("Diff list shape = ", diffList.shape)

                diffList = np.multiply(diffList, self.learningRate)
                dataArrayReshaped = dataArray[i].reshape(1, 785)
                deltaWeight = np.dot(diffList, dataArrayReshaped)
                #print("Delta weight shape = ", deltaWeight.shape)

                self.weights = np.add(self.weights, deltaWeight)

            self.predList.append(np.argmax(np.array(percepValueList)))

        accuracy = ((np.array(self.predList) == np.array(self.actualList)).sum() / float(len(self.actualList))) * 100

        return accuracy

    ### store_accuracy function: used to store accuracy for each learning rate for either validation/train dataset
    ### into respective csv files
    def store_accuracy(self, filepath, accuracyIndex, accuracy):
        with open(filepath, 'a', newline="") as fd:
            csvWriter = csv.writer(fd)
            csvWriter.writerow([accuracyIndex, accuracy])


    ### Loop through training data for 50 epochs and calcualte the accuracy
    def execute_trained_data(self, epoch):
        accuracy = self.learn_and_find_accuracy(epoch, self.trainingArray, self.trainingTargetClassList, True)
        self.trainingAccuracy.append(accuracy)
        self.store_accuracy('train_output' + str(self.learningRate) + '.csv', epoch, accuracy)

    ### Loop through validation data for 50 epochs and calcualte the accuracy
    def execute_validation_data(self, epoch):
        accuracy = self.learn_and_find_accuracy(epoch, self.validationArray, self.validationTargetClassList ,False)
        self.validationAccuracy.append(accuracy)
        self.store_accuracy('validation_output' + str(self.learningRate) + '.csv', epoch, accuracy)
        print("Learning rate = %s, Validation Accuracy for epoch %s = %s" % (self.learningRate, epoch, accuracy))

    ### Loop through training and validation data for 50 epochs and calculate the accuracy
    ###Print confusion matrix for results on test data after 50 epochs of training.
    def learn_digits(self):
        for epoch in range(50):
            self.execute_trained_data(epoch)
            self.execute_validation_data(epoch)
        print("Confusion Matrix: ")
        print(confusion_matrix(self.actualList, self.predList))

if __name__ == "__main__":
    print("Starting time for file load = %s" % (datetime.datetime.now().time()))
    startTime = time.time()
    trainingDataLoader = FileLoader(trainingDataFile)
    trainingDataLoader.load_csv_and_normalize()
    validationDataLoader = FileLoader(validationDataFile)
    validationDataLoader.load_csv_and_normalize()
    endTime = time.time()
    print("End time for file load = %s" % (datetime.datetime.now().time()))
    print("Time taken for loading the files is %s seconds" % (endTime - startTime))

    ### for each learning rate, load the file, train digits.
    for learningRate in (0.1, 0.001, 0.00001):
        print("Starting time for learning rate %s = %s" % (learningRate, datetime.datetime.now().time()))
        startTime = time.time()
        digitIdentifierObj = DigitIdentifier(learningRate)
        digitIdentifierObj.initialize_training_and_validation_file(trainingDataLoader, validationDataLoader)
        digitIdentifierObj.learn_digits()
        endTime = time.time()
        print("Ending time for learning rate %s = %s" % (learningRate, datetime.datetime.now().time()))
        print("Time taken for learning rate %s is %s seconds" % (learningRate, (endTime - startTime)))
        plot_graph(learningRate)
        print("Plotted the graph for learning rate = ", learningRate)