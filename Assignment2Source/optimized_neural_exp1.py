import csv
import numpy as np
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import datetime
import time
import scipy.special
import itertools
import sys, argparse

trainingDataFile = "./mnist_train.csv"
validationDataFile = "./mnist_validation.csv"
learningRate = 0.1

##plotting the graph for training and validation accuracy over epochs
def plot_graph(hiddenNodes):
    trainFile = 'train_output_exp1_' + str(hiddenNodes) + '.csv'
    validationFile = 'validation_output_exp1_' + str(hiddenNodes) + '.csv'

    trainX, trainY = np.loadtxt(trainFile, delimiter=',', unpack=True)
    validationX, validationY = np.loadtxt(validationFile, delimiter=',', unpack=True)

    plt.plot(trainX, trainY, label="Training Set - Experiement 1")
    plt.plot(validationX, validationY, label="Validation Set - Experiement 1")

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%) ')
    plt.title('For hidden nodes ' + str(hiddenNodes))
    plt.legend()
    plt.savefig('graph_exp1_' + str(hiddenNodes) + '.png')
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

class MultiLayerDigitIdentifier:
    def __init__(self, hiddenNodes):
        self.hiddenNodes = hiddenNodes

        self.outputNodes = 10
        self.inputNodes = 785
        self.learningRate = 0.1
        self.epochs = 50
        
        self.wInputToHidden = np.random.uniform(-0.05, 0.05, (self.inputNodes, self.hiddenNodes))
        self.wHiddenToOutput = np.random.uniform(-0.05, 0.05, (self.hiddenNodes + 1, self.outputNodes))
           
        self.correctLabel = []   
        self.predictedLabel = []

        self.hiddenWithBias = np.zeros((1, self.hiddenNodes+1))
        self.hiddenWithBias[0, 0] = 1

        self.trainingAccuracy = []
        self.validationAccuracy = []

        self.activationFunc = lambda x : scipy.special.expit(x)

    def initialize_training_and_validation_file(self, trainingDataLoader, validationDataLoader):
        self.trainingArray = trainingDataLoader.dataArray
        self.trainingTargetClassList = trainingDataLoader.targetClassList

        self.validationArray = validationDataLoader.dataArray
        self.validationTargetClassList = validationDataLoader.targetClassList

    def fill_target_list(self, targetClass):
        targetList = np.zeros((1,self.outputNodes))+0.1
        targetList[0, targetClass] = 0.9
        return targetList

    def learn_and_find_accuracy(self, epoch, dataArray, targetClassList, trainingExecution):
        self.predList = []
        self.actualList = []

        for i in range(0, dataArray.shape[0]):
            targetClass = targetClassList[i]
        
            inputArray = dataArray[i]
            inputArray = inputArray.reshape(1, self.inputNodes)
            #print("Input array shape = ", inputArray.shape)
            #print("W Input to hidden shape = ", self.wInputToHidden.shape)
            hiddenLayerValues = np.dot(inputArray, self.wInputToHidden)
            hiddenLayerSigmoid = self.activationFunc(hiddenLayerValues)
            #print ("hidden outputs shape = ", hiddenLayerSigmoid.shape)

            self.hiddenWithBias[0, 1:] = hiddenLayerSigmoid

            finalLayerValues = np.dot(self.hiddenWithBias, self.wHiddenToOutput)
            finalLayerSigmoid = self.activationFunc(finalLayerValues)
            #print("Final outputs shape = ", finalLayerSigmoid.shape)

            self.predList.append(np.argmax(finalLayerSigmoid))
            self.actualList.append(targetClass)

            if trainingExecution and epoch > 0:
                targetList = self.fill_target_list(targetClass)

                errorOutputLayer = finalLayerSigmoid * (1 - finalLayerSigmoid) * (targetList - finalLayerSigmoid)
                errorHiddenLayer = hiddenLayerSigmoid * (1 - hiddenLayerSigmoid) * np.dot(errorOutputLayer, self.wHiddenToOutput[1:,:].T)

                self.wHiddenToOutput = self.wHiddenToOutput + (self.learningRate * errorOutputLayer * self.hiddenWithBias.T)
                self.wInputToHidden = self.wInputToHidden + (self.learningRate * errorHiddenLayer * inputArray.T)

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
        self.store_accuracy('train_output_exp1_' + str(self.hiddenNodes) + '.csv', epoch, accuracy)

    ### Loop through validation data for 50 epochs and calcualte the accuracy
    def execute_validation_data(self, epoch):
        accuracy = self.learn_and_find_accuracy(epoch, self.validationArray, self.validationTargetClassList ,False)
        self.validationAccuracy.append(accuracy)
        self.store_accuracy('validation_output_exp1_' + str(self.hiddenNodes) + '.csv', epoch, accuracy)
        print("Hidden nodes = %s, Validation Accuracy for epoch %s = %s" % (self.hiddenNodes, epoch, accuracy))

    ### Loop through training and validation data for 50 epochs and calculate the accuracy
    ###Print confusion matrix for results on test data after 50 epochs of training.
    def learn_digits(self):
        for epoch in range(50):
            self.SSE = 0
            self.execute_trained_data(epoch)
            self.execute_validation_data(epoch)
        print("Confusion Matrix: ")
        print(confusion_matrix(self.actualList, self.predList))

if __name__ == "__main__":
    ##Main Method

    print("Starting time for file load = %s" % (datetime.datetime.now().time()))
    startTime = time.time()
    trainingDataLoader = FileLoader(trainingDataFile)
    trainingDataLoader.load_csv_and_normalize()
    validationDataLoader = FileLoader(validationDataFile)
    validationDataLoader.load_csv_and_normalize()
    endTime = time.time()
    print("End time for file load = %s" % (datetime.datetime.now().time()))
    print("Time taken for loading the files is %s seconds" % (endTime - startTime))


    # Create instance of Neural Network
    for hiddenNodes in (20, 50, 100):
        print("Starting time for hidden nodes %s = %s" % (hiddenNodes, datetime.datetime.now().time()))
        startTime = time.time()
        digitIdentifierObj = MultiLayerDigitIdentifier(hiddenNodes)
        digitIdentifierObj.initialize_training_and_validation_file(trainingDataLoader, validationDataLoader)
        digitIdentifierObj.learn_digits()
        endTime = time.time()
        print("Ending time for hidden nodes %s = %s" % (hiddenNodes, datetime.datetime.now().time()))
        print("Time taken for hidden nodes %s is %s seconds" % (hiddenNodes, (endTime - startTime)))
        plot_graph(hiddenNodes)
        print("Plotted the graph for hidden nodes = ", hiddenNodes)
