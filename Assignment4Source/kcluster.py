import numpy as np
import random as r
from collections import Counter
from operator import itemgetter
#
# Casey Nold; nold@pdx.edu; CS545 HW5
# Implementation of K-means clustering
# To run: simply replace the file and test file with the path
# on your machine for the respective files.


def main():
    
    file = '/Users/divya/Documents/Machine Learning/Assignments/Assignment4Source/optdigits.train'
    testFile = '/Users/divya/Documents/Machine Learning/Assignments/Assignment4Source/optdigits.test'

    
    matrix = read(file)
    column, j = matrix.shape
    testMatrix = read(testFile)
    n,o = testMatrix.shape
    NUM_CENTERS = 5

    bestSSE = []; bestSSS =[]; bestEntropy = []; centerList = []

    for i in range(0,1):
        sse, entropy, sss, centers = kmeans(matrix, NUM_CENTERS)
        bestSSE.append(sse); bestSSS.append(sss); bestEntropy.append(entropy); centerList.append(centers)
    
    testPoints = np.argmin(bestSSE)
    testCenters = centerList[testPoints]
    testClusters = partitionData(testMatrix, testCenters)
    print("Test: Sum square sep:", sumSquaredSeparation(testCenters))
    print("Test: Entropy: ",meanEntropy(testClusters,NUM_CENTERS, column))
    print("Test: Sum squared Error:",sumSquaredError(testCenters,testClusters))
    frequency(testClusters, testCenters, n)
    writeMe(testCenters)
    
 
#+-----------------------------------------------------------------------------------------+
# Functions below:                                                                         |
#+-----------------------------------------------------------------------------------------+

def frequency(clusters, centers, dataNum):
    """Take a list of lists of lists, a list of lists and the number of data items. Find
        the frequency that each value in the list of list of lists occurs and display to the
        user. Return the most frequent items."""

    mostFreqPerCluster = []; accuracy = 0
    for each in range(0,len(clusters)):
        count =0
        classes =[]; freq = []
        cluster = clusters.pop(each)
        print("Cluster: ", each)
        for i in cluster:
            classes.append(int(i[-1]))
        classFreq = Counter(classes)
        uniqueItems = classFreq.items()
        for i,j in uniqueItems:
            print("Item:", i, "Frequency:", j)
            count += j
            freq.append((i,j))
        if freq:
            frequent = max(freq,key=itemgetter(1))[1]
            accuracy += frequent
            item = max(freq,key=itemgetter(1))[0]
            print("I'm the most frequent:", frequent, item, count)
            mostFreqPerCluster.append(item)
        else:
            print("Cluster ", each, "is empty")
            mostFreqPerCluster.append(None)
        clusters.insert(each, cluster)
    print("Accuracy:", (accuracy/dataNum))

    return mostFreqPerCluster




def kmeans(data, k):
    """Take a matrix of data and a number of cluster centers. Then run the k-means algorithm on the
        matrix. Return the sum squared error (SSE), sum squared seperation,(SSS), entropy and the centers."""
    sseRun = [];sssRun = [];centerList=[];entropyRun = [];
    i,j = data.shape
    centers = centroid(k)
    for loop in range(0,5):
        clusters = partitionData(data, centers)
        sss = sumSquaredSeparation(centers)
        mec = meanEntropy(clusters,k, i)
        sse = sumSquaredError(centers,clusters)
        entropyRun.append(mec)
        sseRun.append(sse)
        sssRun.append(sss)
        centerList.append(centers)
        centers = updateCentroids(clusters)
    sse, cen, sss, ent = minimumOfOne(sseRun, sssRun,centerList, entropyRun)
    return sse, cen, sss, ent 

def minimumOfOne(listOne, listTwo, listThree, listFour):
    """Take four lists. Find the index containing the minumum value in listOne, return all the values
        of this index from each list."""
    index = np.argmin(listOne)
    return listOne[index], listFour[index], listTwo[index], listThree[index]
        
def tally(bag):
    """count the number of y's in an tuple (x,y). Return the number of y's"""
    counter = 0
    for x, y in bag:
        counter += y
    return counter

def meanEntropy(clusters,k, dataSize):
    """Take a list of list of lists of clusters, the number of cluster centers and the number of feature vectors
        and calculate the mean entropy of the cluster(MEC). Return the MEC"""
    
    classes = []; entropy = 0; log = 0; coeff = []; clusterSize = []; featureList = []; countClass = 0
    for each in range(0,len(clusters)):
        cluster = clusters.pop(each)
        clusterSize.append(len(cluster))
        for i in cluster:
            classes.append(i[-1])
        countClass =  Counter(classes)
        uniqueItems = countClass.items()
        clusterTotal = tally(uniqueItems)
        for i,j in uniqueItems:
            probability = j/clusterTotal
            log += probability * np.log2(probability)
        coeff.append(-log); log = 0
        clusters.insert(each, cluster)
        
    for i in range(0,len(clusters)):
        entropy += ( (clusterSize[i]/(dataSize))*coeff[i])       
    return entropy

def sumSquaredError(centers, clusters):
    """Take a list of lists of centers and a list of lists of feature vectors and find the sum squared error,(SSE).
        Return the SSE."""

    err = 0; bigErr = 0
    for point in range(0,len(centers)):
        feature = clusters[point]
        for i in feature:
            err += np.square(np.asarray(i[:-1]) - np.asarray(centers[point])).sum(axis=0)
    return err 

def sumSquaredSeparation(clusters):
    """Take a list of lists and find the sum squared separation(SSS). Return the SSS."""
    summation = 0
    for each in range(0,len(clusters)):
        cluster = clusters.pop(each)
        for i in clusters:
            summation += np.square(np.asarray(cluster) - np.asarray(i)).sum(axis=0)
        clusters.insert(each, cluster)
    return summation

def updateCentroids(clusters):
    """Take a list of clusters and update the centers for each cluster. Each cluster contains a list of lists of
        feature vectors. Find the mean of each cluster, and assign the center to this value. Return the new
        center values"""

    newCenters = []; 
    for i in range(0,len(clusters)):
        featureVect = []; newVector = []
        if(len(clusters[i]) > 0):
            for each in clusters[i]:
                minusLabel = each[:-1]
                featureVect.append(list(minusLabel))
            newCenters.append(list(np.mean(featureVect,axis=0,dtype=np.int64))) 
        else:
            for i in range(0,64):
                newVector.append(r.uniform(0,17))
            newCenters.append(newVector)
    return newCenters

def partitionData(matrix, centers):
    """Take a matrix of data and a list of vectors and assign each row of the matrix to the
        nearest vector as determined by the euclidean distance. Return a list of lists containing
        the assignment of each row of the matrix"""
    i,j = matrix.shape
    numCenters = len(centers)
    clusters = [[] for i in range(numCenters)]
    
    for row in range(0,i):
        distanceList = []; 
        distanceList = euclidean(matrix[row], centers)
        centerIndex = np.argmin(distanceList)
        clusters[centerIndex].append(list(matrix[row]))
    return clusters


def euclidean(feature, centerList):
    """Take a feature vector and a list of center vectors and calculate the euclidean distance
       from the feature vector to each center vector. Return a list of distances."""
    
    distances = [];
    attributes = feature[:-1] #remove the label
    for each in range(0,len(centerList)):
        center = centerList.pop(each)
        summation = 0
        for i in range(0,len(attributes)):
            if(len(center) ==0):
                break
            summation += np.square(center[i]-attributes[i])
        distance = np.sqrt(summation)
        distances.append(distance)
        centerList.insert(each,center)
    return distances

def centroid(numCenters):
    """Take an integer and create that many lists containing 64 random integers.
        Return the list of lists of random integers"""
    centers = [];centerList = []
    for each in range(0,numCenters):
        centers = []
        for each in range(0,64):
            centers.append(int(r.uniform(0,17)))
        centerList.append(centers)
    return centerList

def read(file):
    """read a file and turn store in a nxm matrix. Add one to each value to scale the data.
        return a nxm matrix"""

    featVect = []

    with open(file,'r') as f:
        line = f.read().splitlines()

    for each in line:
        splitline = each.split(",")
        for element in splitline:
            featVect.append(element)

    column = int(len(featVect)/65)
    matrix = np.zeros(column*65).reshape(column,65)
    i,j =  matrix.shape

    for a in range(0,i):
        for b in range(0,j):
            matrix[a][b] = int(featVect.pop(0))
    return matrix

def writeMe(Llists):
    """ Writes a list of lists to file without the commas."""
    
    with open('/Users/divya/Documents/Machine Learning/Assignments/Assignment4Source/kMeans.pgm','w' ) as f:
        for each in range(0,len(Llists)):
            val = Llists.pop(each)
            for i in range(0,64): 
                digit = val.pop(i)
                intDigit = int(digit)
                f.write(str(intDigit))
                f.write(" ")
                val.insert(i, intDigit)
            f.write('\n')
            Llists.insert(each,val)

if __name__ == "__main__":
    main()