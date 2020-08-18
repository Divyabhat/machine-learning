import numpy as np
from collections import Counter
from scipy.misc import imread, imsave, imresize

class K_Means(object):
    def __init__(self, file_train='optdigits.train', file_test='optdigits.test'):
        # Initialize training and test data
        self.train_x, self.train_y = self.__load_optdigit__(file_train)
        self.test_x, self.test_y = self.__load_optdigit__(file_test)

        # Get the shape of the data
        self.n_samp, self.n_feat = self.train_x.shape
        self.n_test, _ = self.test_x.shape

    def __load_optdigit__(self, filename):
        # Load data from file
        data = np.loadtxt(filename, delimiter=',', unpack=False)
        column = int(len(data)/65)
        # Separate the true values, y, from the data
        y = data[:,-1]
        data = np.delete(data, -1, 1)
        return data, y



    def train_k_means(self, k=10, max_val=16, cent_init='rand'):
        # Save the number of centroids and a placeholder for centroid movement
        self.k = k
        self.delta_cent = np.ones((self.k, self.n_feat))

        # The user can choose to start with random points in the feature space
        # or with the points of randomly selected samples
        if cent_init == 'rand':
            self.centroids = np.random.randint(max_val+1, size=(self.k, self.n_feat))
        else: # cent_init == 'pts':
            ind = np.random.randint(self.n_samp, size=self.k)
            self.centroids = self.train_x[ind]

        # Calculate the location of the cluster centroids
        clusters = self.__find_clusters__(max_iter=200)
        
        # Compute the average mean square error and the mean square separation
        self.avg_mean_sqr_err = self.__calc_avg_mse__(clusters)
        self.mean_sqr_sep = self.__calc_mean_sqr_sep__()
       
        return

    def test_k_means(self):
        # Initialize matrix for holding sample vector coordinate information
        mat = np.zeros((self.n_test, self.k, self.n_feat))

        # Initialize matrix for Euclidean distance from each sample to each centroid
        dist = np.zeros((self.n_test, self.k))

        # Initialize vector to hold the index of the centroid closest to a sample
        d_min = np.zeros(self.n_test)

        # Calculate the coordinates between each sample point and each centroid
        for i in range(self.k):
            mat[:, i, :] = np.subtract(self.centroids[i,:], self.test_x)

        # Compute the distance from each point to each centroid
        dist = np.linalg.norm(mat, axis=2)

        # Calculate the nearest centroid from each point
        d_min = np.argmin(dist, axis=1)

        # Sort the test data values by their nearest centroid
        clusters = [self.test_y[d_min==cent] for cent in range(self.k)]

        # Find the most frequent class of each cluster
        clust_val = [np.bincount(clusters[cent].astype(int), minlength=10) for cent in range(self.k)]

        # Associate a class label to each cluster
        clust_label = [np.argmax(clust_val[cent]) for cent in range(self.k)]

        # Create a confusion matrix
        conf_mat = np.zeros((10, 10))
        for i in range(self.k):
            r = int(clust_label[i])
            conf_mat[r,:] += clust_val[i]

        # Compute the accuracy
        accuracy = np.trace(conf_mat)/np.sum(conf_mat)
        return accuracy, conf_mat 

    def __find_clusters__(self, max_iter=300):
        # Initialize matrix for holding sample vector coordinate information
        mat = np.zeros((self.n_samp, self.k, self.n_feat))

        # Initialize matrix for Euclidean distance from each sample to each centroid
        dist = np.zeros((self.n_samp, self.k))

        # Initialize vector to hold the index of the centroid closest to a sample
        d_min = np.zeros(self.n_samp)

        # Until the centroids stop moving or the max number of iterations is
        # reached, apply k-means algorithm
        for n in range(max_iter): 
            for i in range(self.k):
                mat[:, i, :] = np.subtract(self.centroids[i,:], self.train_x)

            # Compute the distance from each point to each centroid
            dist = np.linalg.norm(mat, axis=2)

            # Calculate the nearest centroid from each point
            d_min = np.argmin(dist, axis=1)

            # Sort the training data by it's nearest centroid
            clusters = [self.train_x[d_min==cent] for cent in range(self.k)]

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

    def __calc_avg_mse__(self, clusters):
        # Compute the averate mean square error
        m = 0
        for c in range(self.k):
            mat = np.subtract(self.centroids[c], clusters[c])
            dist = np.linalg.norm(mat, axis=1)
            m += np.mean(np.power(dist,2))
        return m/self.k

    def __calc_mean_sqr_sep__(self):
        # Initialize square separation variable
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

    
    def gray_scale_centroid(self, filename='centroids'):
        # Save each centroid as it's visualized equivalent
        for i in range(self.k):
            im = self.centroids[i].reshape(8,8)
            im = imresize(im, int(1600))
            imsave(filename +str(i)+ '.png', im)
        return

if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # ----------------------------- Experiment #1 -----------------------------
    # -------------------------------------------------------------------------

    # Select the number of iterations
    n_iter = 5

    knn10 = K_Means()

    # Run the training method n_iter times and select the best one
    min_avg_mse = float('Inf')
    for _ in range(n_iter):
        knn10.train_k_means(k=10, cent_init='pts')

        # Save the state with the minimum average mean square error
        if knn10.avg_mean_sqr_err < min_avg_mse:
            min_avg_mse = knn10.avg_mean_sqr_err
            min_mss = knn10.mean_sqr_sep
            #min_entropy = knn10.avg_entropy
            min_cent = knn10.centroids

    # Assign the state that had the smallest avg mean square error
    knn10.avg_mean_sqr_err = min_avg_mse
    knn10.mean_sqr_sep = min_mss
    #knn10.avg_entropy = min_entropy
    knn10.centroids = min_cent

    # Compute the accuracy and generate a confusion matrix
    accu, conf_mat = knn10.test_k_means()
    print(np.sum(conf_mat))

    # Save the centroids as gray-scale .png files
    knn10.gray_scale_centroid('k10_clusters')

    # Display the results
    print("Average mean-square error: " + str(knn10.avg_mean_sqr_err))
    print("   Mean-square separation: " + str(knn10.mean_sqr_sep))
    #print("   entropy: " + str(knn10.avg_entropy)) 
    print("                 Accuracy: " + str(accu))
    print("Confusion Matrix: ")
    print(conf_mat)

    # -------------------------------------------------------------------------
    # ----------------------------- Experiment #2 -----------------------------
    # -------------------------------------------------------------------------

    knn30 = K_Means()

    # Run the training method n_iter times and select the best one
    min_avg_mse = float('Inf')
    for _ in range(n_iter):
        knn30.train_k_means(k=30, cent_init='pts')

        # Save the state with the minimum average mean square error
        if knn30.avg_mean_sqr_err < min_avg_mse:
            min_avg_mse = knn30.avg_mean_sqr_err
            min_mss = knn30.mean_sqr_sep
            min_cent = knn30.centroids

    # Assign the state that had the smallest avg mean square error
    knn30.avg_mean_sqr_err = min_avg_mse
    knn30.mean_sqr_sep = min_mss
    knn30.centroids = min_cent

    # Compute the accuracy and generate a confusion matrix
    accu, conf_mat = knn30.test_k_means()
    print(np.sum(conf_mat))

    # Save the centroids as gray-scale .png files
    knn30.gray_scale_centroid('k30_clusters')

    # Display the results
    print("Average mean-square error: " + str(knn30.avg_mean_sqr_err))
    print("   Mean-square separation: " + str(knn30.mean_sqr_sep))
    print("                 Accuracy: " + str(accu))
    print("Confusion Matrix: ")
    print(conf_mat)