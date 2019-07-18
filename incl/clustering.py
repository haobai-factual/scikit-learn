import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import *
from sklearn.metrics.pairwise import haversine_distances


df = pd.read_csv('data.csv')

x_train = df[['lat', 'lng']]
x_train['ts'] = df['ts'] * 1E-8

x_train1 = x_train[:int(len(x_train)/2)]
x_train2 = x_train[int(len(x_train)/2):]

seed = 2018

# =============================================================================
# km = KMeans(
#     n_clusters = 3,
#     init = 'k-means++',
#     n_init = 10,
#     max_iter = 300,
#     tol = 1E-4,
#     precompute_distances = 'auto',
#     verbose = 0,
#     random_state = seed,
#     copy_x = True,
#     n_jobs = None,
#     algorithm = 'full')
# =============================================================================

def spatial_temporal_event_distance(p1, p2, **metric_params):
	temporal_distance = metric_params['alpha'] * abs(p1[2] - p2[2])
	return haversine_distances([p1[:2], p2[:2]])[0][1] + temporal_distance

dbs = DBSCAN(
    eps = 5E-2,
    min_samples = 3,
    metric = spatial_temporal_event_distance,
    metric_params={'alpha': 100},
    algorithm = 'auto',
    leaf_size = 30,
    p = None,
    n_jobs = None)

md = dbs.fit(x_train1)
md = dbs.fit(x_train2)

c_label = md.labels_

df['c_label'] = c_label

# TODO: plot

plt.scatter(df['lat'], df['lng'], c = df['c_label'])
plt.show()

