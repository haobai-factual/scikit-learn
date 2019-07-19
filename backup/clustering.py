import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import haversine_distances
from sklearn.cluster import DBSCAN

###############################################################################
# read data
df = pd.read_csv('data.csv')

df2 = df[df['ingested_date'] == '2019-07-02']
df3 = df[df['ingested_date'] == '2019-07-03']
df4 = df[df['ingested_date'] == '2019-07-04']

x_train = df[['lat', 'lng', 'ts']]

x_train2 = df[df['ingested_date'] == '2019-07-02'][['lat', 'lng', 'ts']]
x_train3 = df[df['ingested_date'] == '2019-07-03'][['lat', 'lng', 'ts']]
x_train4 = df[df['ingested_date'] == '2019-07-04'][['lat', 'lng', 'ts']]

###############################################################################
# main function
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

# md = dbs.fit(x_train)  # batch clustering

# incremental clustering
md = dbs.fit(x_train2)  # ingested_date: 2019-07-02
md = dbs.fit(x_train3)  # ingested_date: 2019-07-03
md = dbs.fit(x_train4)  # ingested_date: 2019-07-04

df['c_label'] = md.labels_
df.to_csv('out.csv')

plt.scatter(df['lng'], df['lat'], c = df['c_label'])
plt.show()
