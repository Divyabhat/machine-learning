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
def plot_graph(momentum):
    trainFile = 'train_output_exp3_' + str(momentum) + '.csv'
    validationFile = 'validation_output_exp3_' + str(momentum) + '.csv'

    trainX, trainY = np.loadtxt(trainFile, delimiter=',', unpack=True)
    validationX, validationY = np.loadtxt(validationFile, delimiter=',', unpack=True)

    plt.plot(trainX, trainY, label="Training Set")
    plt.plot(validationX, validationY, label="Validation Set")

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%) ')
    plt.title('For hidden nodes ' + str(momentum))
    plt.legend()
    plt.savefig('graph_exp3_' + str(momentum) + '.png')
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
    def __init__(self, momentum):
        self.hiddenNodes = 100
        self.outputNodes = 10
        self.inputNodes = 785
        self.learningRate = 0.1
        self.epochs = 50
        self.momentum = momentum
        

        ### Initialize the weight variable of the dimension 10 x 785  and initialize weight fot hidden layer
        ### i.e(101 x 10)
        self.wInputToHidden = np.random.uniform(-0.05, 0.05, (self.inputNodes, self.hiddenNodes))
        self.wHiddenToOutput = np.random.uniform(-0.05, 0.05, (self.hiddenNodes + 1, self.outputNodes))
           
        self.correctLabel = []   
        self.predictedLabel = []

        self.wPreviousHiddenToOutput = np.zeros((self.hiddenNodes+1, self.outputNodes))
        self.wPreviousInputToHidden = np.zeros((self.inputNodes, self.hiddenNodes))

        self.hiddenWithBias = np.zeros((1, self.hiddenNodes+1))
        self.hiddenWithBias[0, 0] = 1

        self.trainingAccuracy = []
        self.validationAccuracy = []

        ##Sigmiod activation function
        self.activationFunc = lambda x : scipy.special.expit(x)

    ## Initilize the training and validation data
    def initialize_training_and_validation_file(self, trainingDataLoader, validationDataLoader):
        self.trainingArray = trainingDataLoader.dataArray
        self.trainingTargetClassList = trainingDataLoader.targetClassList

        self.validationArray = validationDataLoader.dataArray
        self.validationTargetClassList = validationDataLoader.targetClassList

    ## function is to fill the target list
    def fill_target_list(self, targetClass):
        targetList = np.zeros((1,self.outputNodes))+0.1
        targetList[0, targetClass] = 0.9
        return targetList

    ## for each value of hidden unit apply forward and backward propagation and 
    ## calclualte the error terms for output and hidden layer.
    def learn_and_find_accuracy(self, epoch, dataArray, targetClassList, trainingExecution):
        self.predList = []
        self.actualList = []

        for i in range(0, dataArray.shape[0]):
            targetClass = targetClassList[i]
        
            inputArray = dataArray[i]
            inputArray = inputArray.reshape(1, self.inputNodes)

            ##Forward propagate the activation times the weights to each node in the hidden layer.
            hiddenLayerValues = np.dot(inputArray, self.wInputToHidden)
            hiddenLayerSigmoid = self.activationFunc(hiddenLayerValues)
           

            self.hiddenWithBias[0, 1:] = hiddenLayerSigmoid
            
            ## Forward propagate the activations times weights from the hidden layer to the output layer.
            finalLayerValues = np.dot(self.hiddenWithBias, self.wHiddenToOutput)
            finalLayerSigmoid = self.activationFunc(finalLayerValues)
         

            self.predList.append(np.argmax(finalLayerSigmoid))
            self.actualList.append(targetClass)

            if trainingExecution and epoch > 0:
                targetList = self.fill_target_list(targetClass)

                ## Calculate the error terms for output unit
                errorOutputLayer = finalLayerSigmoid * (1 - finalLayerSigmoid) * (targetList - finalLayerSigmoid)

                ## Calculate the error terms for hidden unit
                errorHiddenLayer = hiddenLayerSigmoid * (1 - hiddenLayerSigmoid) * np.dot(errorOutputLayer, self.wHiddenToOutput[1:,:].T)

                ## Update the weights for hidden to output layer by introducing the momentum, 
                ## in which change in weight is depend on past weight change
                deltaHiddenWeight = (self.learningRate * errorOutputLayer * self.hiddenWithBias.T) + (self.momentum * self.wPreviousHiddenToOutput)
                self.wPreviousHiddenToOutput = deltaHiddenWeight
                self.wHiddenToOutput = self.wHiddenToOutput + deltaHiddenWeight

                ## Update the weights for input to hidden layer by introducing the momentum,
                ## in which change in weight is depend on past weight change
                deltaInputWeight = (self.learningRate * errorHiddenLayer * inputArray.T) + (self.momentum * self.wPreviousInputToHidden)
                self.wPreviousInputToHidden = deltaInputWeight
                self.wInputToHidden = self.wInputToHidden + deltaInputWeight

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
        self.store_accuracy('train_output_exp3_' + str(self.momentum) + '.csv', epoch, accuracy)

    ### Loop through validation data for 50 epochs and calcualte the accuracy
    def execute_validation_data(self, epoch):
        accuracy = self.learn_and_find_accuracy(epoch, self.validationArray, self.validationTargetClassList ,False)
        self.validationAccuracy.append(accuracy)
        self.store_accuracy('validation_output_exp3_' + str(self.momentum) + '.csv', epoch, accuracy)
        print("Momentum = %s, Validation Accuracy for epoch %s = %s" % (self.momentum, epoch, accuracy))

    ## Loop through training and validation data for 50 epochs and calculate the accuracy
    ## Print confusion matrix for results on test data after 50 epochs of training.
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
    # Vary the momentum value and train the training set,change the weights,and calclualte the accuracy on
    # training set and test for plot.
    # vary numer of hidden units
    for momentum in (0.25, 0.5, 0.95):
        print("Starting time for momentum %s = %s" % (momentum, datetime.datetime.now().time()))
        startTime = time.time()
        digitIdentifierObj = MultiLayerDigitIdentifier(momentum)
        digitIdentifierObj.initialize_training_and_validation_file(trainingDataLoader, validationDataLoader)
        digitIdentifierObj.learn_digits()
        endTime = time.time()
        print("Ending time for momentum %s = %s" % (momentum, datetime.datetime.now().time()))
        print("Time taken for momentum %s is %s seconds" % (momentum, (endTime - startTime)))
        plot_graph(momentum)
        print("Plotted the graph for momentum = ", momentum)
