import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import DBSCAN as  skDBSCAN
from sklearn.datasets import make_moons, make_circles
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import completeness_score


epsilon = 15
min_points = 8
filepath = "User_Data.csv"
X_user_data = np.array(pd.read_csv(filepath))


####K Means Search####
# Sum_of_squared_distances = []
# K = range(5, 25)
# for num_clusters in K :
#     kmeans = sklearn.cluster.KMeans(n_clusters=num_clusters)
#     kmeans.fit(X_moons)
#     Sum_of_squared_distances.append(kmeans.inertia_)
# plt.plot(K,Sum_of_squared_distances,'bx-')
# plt.xlabel('Values of K') 
# plt.ylabel('Sum of squared distances/Inertia') 
# plt.title('Optimal k')
# plt.show()

##### Kmeans 

kmeans_ud_model = sklearn.cluster.KMeans(n_clusters = 2)
kmeans_uds = kmeans_moon_model.fit(X_user_data)


####################################    Graphing     #################################### 

plt.figure(figsize = (20, 4), dpi =80)

# Graphing Moons #
plt.subplot(151)
plt.scatter(x= X_moons[:,0], y= X_moons[:,1], c = custom_dbscan_moons)
plt.title('Custom DBSCAN Moon')
plt.subplot(152)
plt.title('Sklearn DBSCAN Moon')
plt.scatter(x= X_moons[:,0], y= X_moons[:,1], c = sk_dbscan_moons.labels_)
plt.subplot(153)
plt.scatter(x= X_moons[:,0], y= X_moons[:,1], c = kmeans_moons.labels_)
plt.title('Sklearn K-Means Moon (K = 2)')
plt.subplot(154)
plt.scatter(x= X_moons[:,0], y= X_moons[:,1], c = affinity_moons.labels_)
plt.title('Affinity Propagation Moon')
plt.subplot(155)
plt.scatter(x= X_moons[:,0], y= X_moons[:,1], c = mean_moons.labels_)
plt.title('Mean Shift Moon')
plt.tight_layout()
plt.show()
# # Graphing Circles #
# plt.subplot(323)
# plt.scatter(x= X_circles[:,0], y= X_circles[:,1], c = custom_dbscan_circles)
# plt.title('Custom DBSCAN Circles')
# plt.subplot(324)
# plt.title('Sklearn DBSCAN Circles')
# plt.scatter(x= X_circles[:,0], y= X_circles[:,1], c = sk_dbscan_circles.labels_)
plt.figure(figsize = (20, 4), dpi =80)
plt.subplot(151)
plt.scatter(x= X_circles[:,0], y= X_circles[:,1], c = custom_dbscan_circles)
plt.title('Custom DBSCAN Circles')
plt.subplot(152)
plt.title('Sklearn DBSCAN Circles')
plt.scatter(x= X_circles[:,0], y= X_circles[:,1], c = sk_dbscan_circles.labels_)
plt.subplot(153)
plt.scatter(x= X_circles[:,0], y= X_circles[:,1], c = kmeans_circles.labels_)
plt.title('Sklearn K-Means Circles (K = 2)')
plt.subplot(154)
plt.scatter(x= X_circles[:,0], y= X_circles[:,1], c = affinity_circles.labels_)
plt.title('Affinity Propagation Circles')
plt.subplot(155)
plt.scatter(x= X_circles[:,0], y= X_circles[:,1], c = mean_circles.labels_)
plt.title('Mean Shift Circles')
plt.tight_layout()
plt.show()
# # Graphing User Data #
# plt.subplot(325)
# plt.scatter(X_user_data[:,0], X_user_data[:,1],X_user_data[:,2],  c = custom_dbscan_user_data)
# plt.title('Custom DBSCAN User Data')
# plt.subplot(326)
# plt.title('Sklearn DBSCAN User Data')
# plt.scatter(X_user_data[:,0], X_user_data[:,1],X_user_data[:,2], c = sk_dbscan_user_data.labels_)
plt.figure(figsize = (20, 4), dpi =80)
plt.subplot(151)
plt.scatter(x= X_circles[:,0], y= X_circles[:,1], c = custom_dbscan_circles)
plt.title('Custom DBSCAN Circles')
plt.subplot(152)
plt.title('Sklearn DBSCAN Circles')
plt.scatter(x= X_circles[:,0], y= X_circles[:,1], c = sk_dbscan_circles.labels_)
plt.subplot(153)
plt.scatter(x= X_circles[:,0], y= X_circles[:,1], c = kmeans_uds.labels_)
plt.title('Sklearn K-Means Circles (K = 2)')
plt.subplot(154)
plt.scatter(x= X_circles[:,0], y= X_circles[:,1], c = affinity_circles.labels_)
plt.title('Affinity Propagation Circles')
plt.subplot(155)
plt.scatter(x= X_circles[:,0], y= X_circles[:,1], c = mean_circles.labels_)
plt.title('Mean Shift Circles')
plt.tight_layout()
plt.show()
####################################    Printing Accuracy     #################################### 


print("moons dataset")
print("homogeneity score:  ", homogeneity_score(sk_dbscan_moons.labels_,custom_dbscan_moons))
print("rand score:         ", rand_score(sk_dbscan_moons.labels_,custom_dbscan_moons))
print("completeness score: ", completeness_score(sk_dbscan_moons.labels_,custom_dbscan_moons))
print("\n")

print("circles dataset")
print("homogeneity score:  ", homogeneity_score(sk_dbscan_circles.labels_,custom_dbscan_circles))
print("rand score:         ", rand_score(sk_dbscan_circles.labels_,custom_dbscan_circles))
print("completeness score: ", completeness_score(sk_dbscan_circles.labels_,custom_dbscan_circles))
print("\n")

print("userdata dataset")
print("homogeneity score:  ", homogeneity_score(sk_dbscan_user_data.labels_,custom_dbscan_user_data))
print("rand score:         ", rand_score(sk_dbscan_user_data.labels_,custom_dbscan_user_data))
print("completeness score: ", completeness_score(sk_dbscan_user_data.labels_,custom_dbscan_user_data))






####################################    Adding your own Dataset     #################################### 


def dbscan_my_dataset():

	print("Do you want to show dbscan your own dataset. yes or no? ")

	x = input("My answer: ")
	
	if x == "yes":
		

		print("We will ask for your filepath, please ensure all columns are numbers")
		filepath = input("filepath: ")
		### Choose your own constants (edit) ###
		epsilon = 15   #choose your own
		min_points = 8 #choose your own


		### Choose your file (edit) ###
		# filepath = "User_Data.csv" #note all values must be numbers

		### Performing DBSCAN (no need to edit) ###
		X_input = np.array(pd.read_csv(filepath))
		custom_dbscan_input = DBSCAN(epsilon,min_points).predict(X_user_data)
		sk_dbscan_input = skDBSCAN(eps=epsilon, min_samples=min_points).fit(X_user_data)


		### Graphing DBSCANs (no need to edit if your graph is 2d) ###

		#to graph, you must edit the number of dimensions you want to see, we have by default a 2d graph
		plt.subplot(121)
		plt.scatter(X_input[:,0], X_input[:,1], c = custom_dbscan_input)
		plt.title('Custom DBSCAN Input')
		plt.subplot(122)
		plt.title('Sklearn DBSCAN Input')
		plt.scatter(X_input[:,0], X_input[:,1], c = sk_dbscan_input.labels_)


		### Printing closeness custom DBSCAN to sklearn DBSCAN (no need to edit if your graph is 2d) ###

		print("input dataset")
		print("homogeneity score:  ", homogeneity_score(sk_dbscan_input.labels_,custom_dbscan_input))
		print("rand score:         ", rand_score(sk_dbscan_input.labels_,custom_dbscan_input))
		print("completeness score: ", completeness_score(sk_dbscan_input.labels_,custom_dbscan_input))


		#showing graph
		plt.show()

dbscan_my_dataset()