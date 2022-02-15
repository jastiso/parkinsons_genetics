import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.io
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, silhouette_score
from tqdm import tqdm
import re
from wordcloud import WordCloud, STOPWORDS

genes = scipy.io.loadmat('./Brain_data_400/gene_coexpression/ParcellatedGeneExpressionLRHemiSchaefer17Network400.mat')
names = pd.read_csv('./data/gene_ids.csv')

# get coexp
exp = genes['LeftHemiParcelExpression']
# remove missing values from brain data
data_idx = [not x for x in np.isnan(sum(exp.T))]
# demean so its easier to get covariance matrix
exp = exp[data_idx, :]

# get clusters
# normalize
exp = normalize(exp.T)
# plot dendrogram
# Use the ward() function
linkage_array = ward(exp)
# Now we plot the dendrogram for the linkage_array containing the distances
# between clusters
dendrogram(linkage_array)
ax = plt.gca()
bounds = ax.get_xbound()
plt.xlabel('Data index')
plt.ylabel('Cluster distance')
plt.show()
plt.tight_layout()
plt.savefig("./img/clusters/dend.jpg")
plt.close()

# assign data to clusters
# n clusters taken from dendrogram
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
cluster.fit_predict(exp)
print(str(sum(cluster.labels_ == 0)))
print(str(sum(cluster.labels_ == 1)))
print(str(sum(cluster.labels_ == 2)))
print(str(sum(cluster.labels_ == 3)))
print(f"SI: {silhouette_score(exp, cluster.labels_)}")
print(f"CH: {calinski_harabasz_score(exp, cluster.labels_)}")

# plot in PCA space
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(exp)
principalDf = pd.DataFrame(data=principalComponents,
                           columns=['principal component 1', 'principal component 2', 'principal component 3'])
plot_df = pd.concat([principalDf, pd.DataFrame({'label': cluster.labels_})], axis=1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = plot_df['principal component 1']
y = plot_df['principal component 2']
z = plot_df['principal component 3']
c = plot_df['label']
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.scatter(x, y, z, c=c)
plt.tight_layout()
plt.savefig("./img/clusters/clusters.jpg")
plt.close()

# check for clusters with scrambled data
# stability
# 10 fold cross validation to recluster
kf = KFold(n_splits=10)
cnt = 0
for train_index, test_index in kf.split(exp):
    X_train = exp[train_index]
    sub_cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
    sub_cluster.fit_predict(X_train)
    score = adjusted_rand_score(sub_cluster.labels_, cluster.labels_[train_index])
    # plot in PCA space
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(X_train)
    principalDf = pd.DataFrame(data=principalComponents,
                               columns=['principal component 1', 'principal component 2', 'principal component 3'])
    plot_df = pd.concat([principalDf, pd.DataFrame({'label': sub_cluster.labels_})], axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = plot_df['principal component 1']
    y = plot_df['principal component 2']
    z = plot_df['principal component 3']
    c = plot_df['label']
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(str(score))
    scatter = ax.scatter(x, y, z, c=c, label=str(c))
    handles, labels = scatter.legend_elements(prop="colors")
    legend = ax.legend(*scatter.legend_elements(), loc="upper right")
    plt.tight_layout()
    plt.savefig(f"./img/clusters/kfold/clusters{cnt}.jpg")
    plt.close()
    cnt += 1

# significance
nPerm = 500
si = []
ch = []
for p in tqdm(range(nPerm)):
    # shuffle indices relative to eachother
    X = exp.copy()
    for r in range(np.shape(X)[0]):
        np.random.shuffle(X[r, :])

    # cluster
    perm_cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
    perm_cluster.fit_predict(X)

    # get SI and CH scores
    si.append(silhouette_score(exp, perm_cluster.labels_))
    ch.append(calinski_harabasz_score(exp, perm_cluster.labels_))

# plot
sns.histplot(si)
plt.axvline(silhouette_score(exp, cluster.labels_), color='r', linestyle='dashed', linewidth=1)
plt.tight_layout()
plt.savefig(f"./img/clusters/si.jpg")
plt.close()

sns.histplot(ch)
plt.axvline(calinski_harabasz_score(exp, cluster.labels_), color='r', linestyle='dashed', linewidth=1)
plt.tight_layout()
plt.savefig(f"./img/clusters/ch.jpg")
plt.close()

# if we sample from these clusters, do we get good estimates of the corr matrix
nIter = 500
nGenes = 50
ids = list(range(np.shape(exp)[1]))
A = np.corrcoef(exp)

# sample from clusters
clusters_sim = []
ns = np.round((pd.DataFrame(cluster.labels_).value_counts() / len(cluster.labels_)) * nGenes).values
for i in tqdm(range(nIter)):
    curr_idx = []
    for c in np.unique(cluster.labels_):
        curr_idx.extend(np.random.choice([x for x in ids if cluster.labels_[x] == c], int(ns[c]), replace=False))
    samp_exp = np.corrcoef(exp[:, curr_idx])
    clusters_sim.append(np.linalg.norm(A - samp_exp))
# sample from the same cluster
same_sim = []
for i in tqdm(range(nIter)):
    curr_cluster = np.random.choice(list(np.unique(cluster.labels_)), 1)
    curr_idx = np.random.choice([x for x in ids if cluster.labels_[x] == curr_cluster[0]], nGenes, replace=False)
    samp_exp = np.corrcoef(exp[:, curr_idx])
    same_sim.append(np.linalg.norm(A - samp_exp))

# plot
bins = np.linspace(0, np.max(same_sim) + 5, 100)
plt.hist(clusters_sim, bins, alpha=0.5, label='cluster')
plt.hist(same_sim, bins, alpha=0.5, label='same', color='grey')
plt.legend()
plt.xlabel('count')
plt.ylabel('||A - Ahat||')
plt.tight_layout()
plt.savefig(f"./img/clusters/matrix_dist.jpg")
plt.close()

# function of clusters
for c in np.unique(cluster.labels_):
    syms = names.ID[cluster.labels_ == c].values
    functions = [genes.loc[x, 'Biological'] for x in genes.index if genes.loc[x, 'symbol'] in syms]
    # cleaning
    functions = [x for x in functions if not pd.isnull(x)]
    functions = [re.sub("[\(\[].*?[\)\]]", "", x) for x in functions]
    functions = [x.replace(";", " ") for x in functions]
    function = " ".join(functions)
    # word cloud
    wc = WordCloud(background_color='white',stopwords=set(STOPWORDS)).generate(function)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(f"./img/clusters/wc{c}.jpg")
    plt.close()
