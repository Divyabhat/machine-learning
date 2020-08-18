import numpy as np
from collections import Counter
from scipy.misc import imread, imsave, imresize

number_of_clusters = 10

class K_Means(object):
    def __init__(self, train_file='optdigits.train', test_file='optdigits.test'):
        # Initialize training and test data
        self.train_data, self.train_labels = self.load_data(train_file)
        self.test_data, self.test_labels = self.load_data(test_file)

        # Get the shape of the data
        self.n_samp, self.n_feat = self.train_data.shape
        self.n_test, _ = self.test_data.shape

 
    def load_data(self,file1):
        data = np.loadtxt(file1, delimiter=',',unpack=False)
        
        #store class labels and data seperatly

        labels = data[:,-1]
        data = np.delete(data,-1,1)
        return data,labels


    def kmeans_train(self, k =10,max_val =16,cent_flag = 'randm'):
        i, j = self.train_data.shape
        self.k = k
        self.delta_cent = np.ones((self.k, self.n_feat))

        #choose the random point in the future space or 
        #randomly selected samples

        if cent_flag == 'randm':
           self.centroids = np.random.randint(max_val+1, size=(self.k, self.n_feat))
        else:
            ind = np.random.randint(self.n_samp, size=self.k)
            self.centroids = self.train_data[ind]

        #location of cluster centroids
        clusters = self.find_clusters(max_itr = 200)

        # Compute the average MSE and MSS and Entropy
        self.mse_avg = self.calculate_avg_mse(clusters)
        self.mss_avg = self.calculate_avg_mss()
        self.mean_entropy = self.calculate_entropy(clusters,k,i)
        return
    

    def kmeans_test(self):
        # holding sample vector coordinate information
        sam_matrix = np.zeros((self.n_test, self.k, self.n_feat))

        # Euclidean distance from each sample to each centroid
        elu_dist = np.zeros((self.n_test, self.k))

        # index of the centroid closest to a sample
        min_dis_centroid = np.zeros(self.n_test)

        #coordinates between each sample point and each centroid
        for i in range(self.k):
            sam_matrix[:, i, :] = np.subtract(self.centroids[i,:], self.test_data)

        #distance from each point to each centroid
        elu_dist = np.linalg.norm(sam_matrix, axis=2)

        #nearest centroid from each point
        min_dis_centroid = np.argmin(elu_dist, axis=1)

        # Sort the test data values by their nearest centroid
        clusters = [self.test_labels[min_dis_centroid==cent] for cent in range(self.k)]

        # Find the most frequent class of each cluster
        clust_val = [np.bincount(clusters[cent].astype(int), minlength=10) for cent in range(self.k)]

        # Associate a class label to each cluster
        clust_label = [np.argmax(clust_val[cent]) for cent in range(self.k)]

        # Create a confusion matrix
        confusion_matrix = np.zeros((10, 10))
        for i in range(self.k):
            r = int(clust_label[i])
            confusion_matrix[r,:] += clust_val[i]

        # Accuracy
        accuracy = np.trace(confusion_matrix)/np.sum(confusion_matrix)
        return accuracy, confusion_matrix 


    def find_clusters(self, max_itr=300):
        # sample vector coordinate information
        sample_matrix = np.zeros((self.n_samp, self.k, self.n_feat))

        # Euclidean distance from each sample to each centroid
        elu_dist = np.zeros((self.n_samp, self.k))

        # Initialize vector to hold the index of the centroid closest to a sample
        min_dist = np.zeros(self.n_samp)

        # Until the centroids stop moving or the max number of iterations is
        # reached, apply k-means algorithm
        for n in range(max_itr): 
            for i in range(self.k):
                sample_matrix[:, i, :] = np.subtract(self.centroids[i,:], self.train_data)

            # Compute the distance from each point to each centroid
            elu_dist = np.linalg.norm(sample_matrix, axis=2)

            # Calculate the nearest centroid from each point
            min_dist = np.argmin(elu_dist, axis=1)

            # Sort the training data by it's nearest centroid
            clusters = [self.train_data[min_dist==cent] for cent in range(self.k)]

            for i in range(self.k):
                if len(clusters[i] > 0):
                    temp = np.mean(clusters[i], axis=0)
                    self.delta_cent[i] = np.subtract(self.centroids[i], temp)
                    self.centroids[i] = temp

            moving = np.count_nonzero(self.delta_cent)
            if not moving:
                print("Clustering stopped after " + str(n) + " iterations")
                break
            
        return clusters

    # Calculate average mean square error
    def calculate_avg_mse(self,clusters):
        m = 0
        for c in range(self.k):
            mat = np.subtract(self.centroids[c], clusters[c])
            dist = np.linalg.norm(mat, axis=1)
            m += np.mean(np.power(dist,2))
        return m/self.k


    # Calculate mean square error
    def calculate_avg_mss(self):
        sqr_sep = 0

        # For each pairing of centroids, compute the mean square separation
        i = 0
        count = 0
        while i < self.k:
            j = i + 1
            while j < self.k:
                a = self.centroids[i,:]
                b = self.centroids[j,:]
                sqr_sep += np.power(np.linalg.norm(np.subtract(a,b)), 2)
                count += 1
                j += 1
            i += 1
        return sqr_sep/count

    
    def tally(self,bag):
        counter = 0
        for x, y in bag:
            counter += y
        return counter


    def calculate_entropy(self,clusters,k,datasize):
        
        classes = []; entropy = 0; log = 0; coeff = []; clusterSize = []; featureList = []; countClass = 0
        for each in range(0,len(clusters)):
            cluster = clusters.pop(each)
            print(cluster)
            clusterSize.append(len(cluster))
            print(clusterSize)
            for i in cluster:
                classes.append(i[-1])
            countClass = Counter(classes)
            print(countClass)
            uniqueItems = countClass.items()
            print(uniqueItems)
            clusterTotal = self.tally(uniqueItems)
            for i,j in uniqueItems:
                probability = j/clusterTotal
                log += probability * np.log2(probability)
            coeff.append(-log); log = 0
            clusters.insert(each, cluster)
        
        for i in range(0,len(clusters)):
            entropy += ( (clusterSize[i]/(datasize))*coeff[i])       
        return entropy

   
      

    # Save each centroid as  visualized equivalent
    def gray_scale_centroid(self, filename='centroids'):
        for i in range(self.k):
            im = self.centroids[i].reshape(8,8)
            im = imresize(im, int(1600))
            imsave(filename +str(i)+ '.png', im)
        return



if __name__ == '__main__':

    ## For experiment 1 clusters = 10

    kmeans_cluster_10 = K_Means()
    trails = 5


    ## Run the training method for 5 trails and select the  minimum value
    #  and Save the state with the minimum average MSE

    min_avg_mse = float('Inf')
    for _ in range(trails):
        kmeans_cluster_10.kmeans_train(k=10, cent_flag='pts')

        if kmeans_cluster_10.mse_avg < min_avg_mse:
            min_avg_mse = kmeans_cluster_10.mse_avg
            mss_min = kmeans_cluster_10.mss_avg
            mss_entropy = kmeans_cluster_10.mean_entropy
            min_cent = kmeans_cluster_10.centroids

    # Assign the state that had the smallest avg mean square error
    kmeans_cluster_10.mse_avg = min_avg_mse
    kmeans_cluster_10.mss_avg = mss_min
    #knn10.avg_entropy = min_entropy]
    kmeans_cluster_10.mean_entropy = mss_entropy
    kmeans_cluster_10.centroids = min_cent

    # Compute the accuracy and generate a confusion matrix
    accuracy, confusion_matrix = kmeans_cluster_10.kmeans_test()
    print(np.sum(confusion_matrix))

    # Save the centroids as gray-scale .png files
    kmeans_cluster_10.gray_scale_centroid('k10_clusters')

    # Display the results
    print("Average mean-square error: " + str(kmeans_cluster_10.mse_avg))
    print("   Mean-square separation: " + str(kmeans_cluster_10.mss_avg))
    print("   entropy: " + str(kmeans_cluster_10.mean_entropy)) 
    print("                 Accuracy: " + str(accuracy))
    print("Confusion Matrix: ")
    print(confusion_matrix)

    
    kmeans_cluster_30 = K_Means()

    # Run the training method n_iter times and select the best one
    min_avg_mse = float('Inf')
    for _ in range(trails):
        kmeans_cluster_30.kmeans_train(k=30, cent_flag='pts')

        # Save the state with the minimum average mean square error
        if kmeans_cluster_30.mse_avg < min_avg_mse:
            min_avg_mse = kmeans_cluster_30.mse_avg
            min_mss = kmeans_cluster_30.mss_avg
            min_cent = kmeans_cluster_30.centroids

    # Assign the state that had the smallest avg mean square error
    kmeans_cluster_30.mse_avg = min_avg_mse
    kmeans_cluster_30.mss_avg = min_mss
    kmeans_cluster_30.centroids = min_cent

    # Compute the accuracy and generate a confusion matrix
    accuracy, confusion_matrix = kmeans_cluster_30.kmeans_test()
    print(np.sum(confusion_matrix))

    # Save the centroids as gray-scale .png files
    kmeans_cluster_30.gray_scale_centroid('k30_clusters')

    # Display the results
    print("Average mean-square error: " + str(kmeans_cluster_30.mse_avg))
    print("   Mean-square separation: " + str(kmeans_cluster_30.mss_avg))
    print("                 Accuracy: " + str(accuracy))
    print("Confusion Matrix: ")
    print(confusion_matrix)





