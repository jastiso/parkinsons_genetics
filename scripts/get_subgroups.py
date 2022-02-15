from functions.control_helpers import multilayer_control, add_func
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import scipy.io
from scipy.stats import spearmanr

genes = pd.read_csv("./data/pantherGeneList.txt", sep='\t',
                    names=['ID', 'MappedID', 'Name', 'Protein', 'Biological', 'Molecular'], index_col=False)
bps = pd.read_csv("./data/pantherGeneCats.txt", sep='\t',
                  names=['CatID', 'ID', 'Name', 'Parent', 'Child', 'Family'], index_col=False)
clean_genes = genes.dropna(axis='rows', subset='Biological')

# find base categories
top_idx = ['biological_process' == x for x in bps.Parent]
top = bps.loc[top_idx, 'Name']

# merge tables recursively
for c in top:
    name = c
    add_func(c, name, bps, genes)

genes.info()

# plot
plot_df = genes.melt(value_vars=top).dropna(subset='value', axis='rows')
sns.countplot(data=plot_df, x='variable', palette='Set3')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
plt.savefig("./img/subgrp_coexp/subgrps.jpg")
plt.close()

# save
# add gene symbol for easier relating
genes = genes.assign(symbol=genes.Name.apply(lambda x: x.split(";")[1]))
genes.to_csv("./data/gene_list_cats.csv")

# make gene co-exp matrices for top categories
dat = scipy.io.loadmat('./Brain_data_400/gene_coexpression/ParcellatedGeneExpressionLRHemiSchaefer17Network400.mat')
exp = dat['LeftHemiParcelExpression']
names = [x[0][0] for x in dat['gene_names']]

# remove missing values from brain data
data_idx = [not x for x in np.isnan(sum(exp.T))]
exp = exp[data_idx, :]

# loop through subgroups and save matrices
# get biggest groups
top = plot_df.variable.value_counts().index[plot_df.variable.value_counts() > 100].values
top = np.append(top, ['all'])
for c in top:
    if c == 'all':
        syms = genes['symbol'].unique()
    else:
        syms = genes.loc[genes[c] > 0, 'symbol'].unique()
    curr_idx = [x in syms for x in names]
    curr_exp = exp[:, curr_idx]
    # make corr matrix
    A = np.corrcoef(curr_exp)
    A = A - np.eye(np.shape(A)[0])
    # plot
    plt.figure()
    sns.heatmap(A)
    plt.savefig(f"./img/subgrp_coexp/A_{c.split(' ')[0]}.jpg")
    plt.close()
    # save
    np.save(f'./data/subgrp_coexp/A_{c.split(" ")[0]}.npy', A)
    if c == 'all':
        np.savetxt(f'./data/subgrp_coexp/A_{c.split(" ")[0]}.csv', A, delimiter=',')

# multilayer control
# get FC and index
fc = pd.read_csv("./Brain_data_400/Aver_REST1_FC.csv", header=None).to_numpy()
fc = fc[data_idx, :]
fc = fc[:, data_idx]
np.savetxt(f'./data/subgrp_coexp/A_fn.csv', fc, delimiter=',')

energies = pd.DataFrame(columns=['j', 'g_tar_u', 'f_tar_u', 'g_err', 'f_err', 'subgroup'])
for c in tqdm(top):
    gc = np.load(f'./data/subgrp_coexp/A_{c.split(" ")[0]}.npy')

    # get energy
    T = 5
    g_tar_u, g_tar_err = multilayer_control(fc, gc, T)
    f_tar_u, f_tar_err, = multilayer_control(gc, fc, T)

    # add to energy
    curr_eng = pd.DataFrame({
        'j': list(range(np.shape(gc)[0])),
        'g_tar_u': np.mean(g_tar_u, axis=1),
        'f_tar_u': np.mean(f_tar_u, axis=1),
        'g_err': np.log10(g_tar_err),
        'f_err': np.log10(f_tar_err),
        'subgroup': c
    })
    energies = pd.concat([energies, curr_eng])

energies.set_index(pd.Index(list(range(np.shape(gc)[0]*np.size(top)))), inplace=True)


# make useful columns
energies = energies.assign(
    gf_diff=energies.g_tar_u - energies.f_tar_u
)
energies.to_csv("./data/subgrp_coexp/energies.csv")

# plot
# error
plt.figure()
sns.histplot(data=energies, x='g_err', hue='subgroup')
plt.tight_layout()
plt.savefig("./img/subgrp_coexp/g_err.jpg")
plt.close()

plt.figure()
sns.histplot(data=energies, x='f_err', hue='subgroup')
plt.tight_layout()
plt.savefig("./img/subgrp_coexp/f_err.jpg")
plt.close()

# energy
plt.figure()
plt_data = energies[energies['subgroup'] == 'cellular process'].melt(id_vars='j',
                                                        value_vars=['g_tar_u', 'f_tar_u'],
                                                        value_name='energy',
                                                        var_name='target')
sns.scatterplot(data=plt_data, x='j', y='energy', hue='target')
plt.tight_layout()
plt.savefig("./img/subgrp_coexp/energy_diff_all.jpg")
plt.close()

plt.figure()
sns.scatterplot(data=energies, x='j', y='gf_diff', hue='subgroup')
plt.tight_layout()
plt.savefig("./img/subgrp_coexp/energy_diff.jpg")
plt.close()

# check relationship between number of genes and energy
#energies = pd.read_csv("./data/subgrp_coexp/energies.csv")
for c in top:
    if c != 'all':
        energies.loc[energies['subgroup'] == c, 'count'] = len(genes.loc[genes[c] > 0, 'symbol'].unique())
    else:
        energies.loc[energies['subgroup'] == c, 'count'] = len(genes['symbol'].unique())
avg_data = energies[['count', 'subgroup', 'gf_diff']].groupby(['subgroup', 'count']).median().reset_index()
spearmanr(avg_data['count'], avg_data['gf_diff'])
plt.figure()
sns.regplot(data=avg_data, x='count', y='gf_diff')
plt.tight_layout()
plt.savefig("./img/subgrp_coexp/energy_diff_size.jpg")
plt.close()

