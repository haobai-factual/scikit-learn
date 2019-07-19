import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import haversine_distances
from sklearn.cluster import *


df = pd.read_csv('data.csv')

df2 = df[df['ingested_date'] == '2019-07-02']
df3 = df[df['ingested_date'] == '2019-07-03']
df4 = df[df['ingested_date'] == '2019-07-04']

x_train = df[['lat', 'lng', 'ts']]

x_train2 = df[df['ingested_date'] == '2019-07-02'][['lat', 'lng', 'ts']]
x_train3 = df[df['ingested_date'] == '2019-07-03'][['lat', 'lng', 'ts']]
x_train4 = df[df['ingested_date'] == '2019-07-04'][['lat', 'lng', 'ts']]


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

def st_distance(p1, p2, **params):
	dt = params['alpha'] * abs(p1[2] - p2[2])
	return haversine_distances([p1[:2], p2[:2]])[0][1] * 1E3 + dt

dbs = DBSCAN(
    eps = 50,
    min_samples = 3,
    metric = st_distance,
    metric_params={'alpha': 0.2},
    algorithm = 'auto',
    leaf_size = 30,
    p = None,
    n_jobs = None)

md = dbs.fit(x_train2)
md = dbs.fit(x_train3)
md = dbs.fit(x_train4)
#md = dbs.fit(x_train)
c_label = md.labels_


df4a = df2.append(df3).append(df4)
df4a['c_label'] = c_label

df4a.to_csv('out4.csv')

# TODO: plot

#plt.scatter(df['lng'], df['lat'], c = df['c_label'])
#plt.show()
