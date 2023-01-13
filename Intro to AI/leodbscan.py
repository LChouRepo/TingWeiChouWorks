import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN as  DBSCAN2
def dist(X,Y):
	#euclidean distance
	return np.sqrt(sum([(x-y)*(x-y) for x,y in zip(X,Y)]))
class DBSCAN():
	def __init__(self, eps=1, minSamples=10):
		self.eps = eps
		self.minSamples = minSamples

	def expand(self, sample, neighbors):
		cluster = [sample]
		for neighbor in neighbors:
			if not neighbor in self.visited:
				self.visited.append(neighbor)
				self.neighbors[neighbor] = self.get_neigs(neighbor)
				if len(self.neighbors[neighbor]) >= self.minSamples:
					expanded_cluster = self.expand(neighbor, self.neighbors[neighbor])
					cluster = cluster + expanded_cluster
				else:
					cluster.append(neighbor)
		return cluster

	def get_neigs(self, sample):
		"""
		return array of neighbors
		"""
		ID = np.arange(len(self.X))
		neighbors = [i   for i,s in enumerate(self.X[ID != sample]) if (dist(self.X[sample], s) <self.eps) ]
		return np.array(neighbors)

	def get_labels(self):
		"""
		assign label of 
		each samples of each cluster
		"""
		labels = np.full(shape=self.X.shape[0], fill_value=len(self.clusters))
		for i, cluster in enumerate(self.clusters):
			for sample in cluster:
				labels[sample] = i
		return labels

	def predict(self, X):
		"""
		based a dataset returns labels to X
		"""
		self.X = X
		self.clusters = []
		self.visited = []
		self.neighbors = {}
		n_samples = np.shape(self.X)[0]
		for sample in range(n_samples):
			if  not sample in self.visited:
				self.neighbors[sample] = self.get_neigs(sample)
				if len(self.neighbors[sample]) >= self.minSamples:
					self.visited.append(sample)
					new_cluster = self.expand(sample, self.neighbors[sample])
					self.clusters.append(new_cluster)

		cluster_labels = self.get_labels()
		return cluster_labels



df=pd.read_csv('Use_Data.csv')
df['Gender']=df['Gender'].map({'Male':0,'Female':1})
df=df.drop(['CustomerID'],axis=1)
scaler = StandardScaler()

X=df.to_numpy()

X=scaler.fit_transform(X)
clf=DBSCAN(0.5,6)
# ans=clf.predict(X)
# clustering = DBSCAN2(eps=0.3, min_samples=1).fit(X)



moonx, moony = datasets.make_moons(100, noise=2, random_state=20)
moondf = pd.DataFrame(moonx, columns=['X', 'Y'])
# moondf['label'] = moony

moonNP = moondf.to_numpy()
moonNP = scaler.fit_transform(moonNP)


ansMoon=clf.predict(moonNP)
clusteringMoon = DBSCAN2(eps=0.5, min_samples=6).fit(moonNP)

print(ansMoon)
print(clusteringMoon.labels)



# moonnp = moon.to_numpy()
# moonnp = scaler.fit_transform(moonnp)

# print(ans)

# print(clustering.labels_)

from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import completeness_score
print("homogeneity score:  ", homogeneity_score(clusteringMoon.labels_,ansMoon))
print("rand score:         ", rand_score(clusteringMoon.labels_,ansMoon))
print("completeness score: ", completeness_score(clusteringMoon.labels_,ansMoon))
