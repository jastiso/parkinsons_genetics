import numpy as np
from scipy.sparse.linalg import eigs
from network_control.energies import minimum_input, integrate_u


def opt_genes(b, lam, X, A):
    """

    :param b: weighting for gene expression
    :param lam: weighting for regulatization
    :param X: gene expression
    :param A: co-expression matrix
    :return: the loss for this value of b and lam
    """
    B = np.corrcoef(X * b)
    err = np.linalg.norm(A - B)
    return err + (lam * np.linalg.norm(b, ord=1))


def multilayer_control(G1, G2, T):
    """
    Setting up multilayer control function
    :param G1: input layer, NxN matrix
    :param G2: target layer, NxN matrix
    :param T: time horizon
    :return: u, n_err: energy, and error
    """
    # constants
    norm_fac, _ = eigs(G1, k=1)
    N = np.shape(G1)[0]

    # make input matrix
    B = np.concatenate((np.eye(N), np.zeros(np.shape(G1))))

    # make multilayer network
    duplex0 = np.concatenate((
        np.concatenate((G1, np.zeros(np.shape(G1))), axis=1),
        np.concatenate((np.eye(N), G2), axis=1)
    ))

    # normalize
    duplex = duplex0 / np.real(norm_fac)

    # get layers
    layer2 = duplex[N:2 * N, N:2 * N]

    # get eigs of each layer
    D2, V2 = np.linalg.eig(layer2)
    I2 = np.argsort(D2)
    V2 = V2[:, I2]

    # get minimum energy
    x0 = np.zeros((2 * N, 1))
    u = np.zeros((1, N))
    n_err = []
    for j_eig in range(N):
        # final states
        xf = np.concatenate((np.zeros((N,)), V2[:, j_eig]))

        # energy
        _, uj, n_errj = minimum_input(duplex, T, B, x0, xf)

        uj = integrate_u(uj)
        uj = np.reshape(uj, (1, N))
        u = np.concatenate((u, uj))
        n_err.append(n_errj)
    # remove empty first u reference
    u = u[1:, :]
    return u, n_err


def add_func(category, g, bps, genes):
    """
    This function dds broad category labels to the genes dataframe
    :param genes: gene dataframe
    :param bps: biological processes dataframe
    :param category: the top level category
    :param g: the specific process label
    :return: None - but modifies genes dataframe
    """
    if sum(bps.Name == g) > 0:
        gene_list = bps.loc[bps.Name == g, 'ID'].values[0].split(',')
        curr_idx = [x in gene_list for x in genes.ID]
        genes.loc[curr_idx, category] = 1
        # go to next level
        if not bps.loc[bps.Name == g, 'Child'].isna().values:
            child_names = bps.loc[bps.Name == g, 'Child'].values[0].split(';')
            for n in child_names:
                add_func(category, n, bps, genes)
