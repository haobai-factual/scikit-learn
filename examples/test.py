from sklearn.cluster import DBSCAN
import numpy as np

X = np.array([[1, 2], [2, 2], [2, 3],
              [8, 7], [8, 8], [25, 80]])
dbscan_model = DBSCAN(eps=3, min_samples=2)
clustering = dbscan_model.fit(X)
print(clustering.labels_)
# [0 0 0 1 1 -1]

Y = np.array([[25,81], [25, 79]])
clustering = dbscan_model.fit(Y)
print(clustering.labels_)
#[0 0 0 1 1 2 2 2]

Z = np.array([[3, 3], [4, 4], [5, 5], [6,6]])
clustering = dbscan_model.fit(Z)
print(clustering.labels_)


from sklearn.metrics.pairwise import haversine_distances

def spatial_temporal_event_distance(p1, p2, **metric_params):
	temporal_distance = metric_params['alpha'] * abs(p1[2] - p2[2])
	print(haversine_distances([p1[:2], p2[:2]])[0][1] + temporal_distance)
	return haversine_distances([p1[:2], p2[:2]])[0][1] + temporal_distance

p1 = [33.943380, -118.408958, 1000000]
p2 = [33.943380, -118.408958, 1000000]

dbscan_model = DBSCAN(eps=5E-2, min_samples=2, metric=spatial_temporal_event_distance, metric_params={'alpha': 1})
print(np.array([p1, p2]))
clustering = dbscan_model.fit(np.array([p1, p2]))
print(clustering.labels_)






