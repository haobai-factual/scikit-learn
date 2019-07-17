import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import *


df = pd.read_csv('data.csv')

x_train = df[['lat', 'lng']]
x_train['ts'] = df['ts'] * 1E-8

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

dbs = DBSCAN(
    eps = 5E-2,
    min_samples = 3,
    metric = 'euclidean',
    metric_params = None,
    algorithm = 'auto',
    leaf_size = 30,
    p = None,
    n_jobs = None)

md = dbs.fit(x_train)

c_label = md.labels_

df['c_label'] = c_label

# TODO: plot

plt.scatter(df['lat'], df['lng'], c = df['c_label'])
