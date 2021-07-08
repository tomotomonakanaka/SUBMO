import numpy as np
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import DataStructs


def greedy_logdet_max(df, percentage=None):
    ''' ranking in terms of maximizing diversity in terms of logdet
    df: dataframe with embedding
    k: ranking n*k molecules
    '''
    X = np.array(list(df["embedding"]))
    n = X.shape[0]

    def L_ii(i):
        return 1 + X[i].dot(X[i])

    def L_Zj(Z, j):
        return X[Z].dot(X[j])

    # greedy by Chen et al. 2018
    d2 = np.array([L_ii(i) for i in range(n)])
    j = np.argmax(d2)
    Z = list(range(n))
    rank = [j]
    Z.remove(j)

    # define k (cardinality constraints)
    if percentage is None:
        k = n
    else:
        k = int(n*percentage)

    # greedy
    while len(rank) < k:
        e = np.ones(n)*np.inf
        if len(rank) == 1:
            e[Z] = L_Zj(Z, j) / np.sqrt(d2[j])
            c = e
        else:
            e[Z] = (L_Zj(Z, j) - c[Z].dot(c[j])) / np.sqrt(d2[j])
            c = np.c_[c, e]
        d2[Z] -= e[Z]**2
        j = Z[np.argmax(d2[Z])]
        rank.append(j)
        Z.remove(j)
    rank = np.append(rank, Z)
    df['logdet_ranking'] = np.argsort(rank)
    df.loc[df['logdet_ranking'] >= k, 'logdet_ranking'] = len(df)

    return df


def greedy_baseline(df, percentage=None, dissimilarity=None, rule="maxsum", vector=""):
    ''' ranking by maximizing maxsum or maxmin
    df: dataframe with embedding
    percentage: ranking n*percentage molecules
    dissimilarity: None, Tanimoto
    rule: maxsum, maxmin
    vector: _maccs
    '''
    if vector == "":
        X = list(df["embedding"])
    elif vector == '_maccs':
        X = list(df['maccs'])
    elif vector == '_ecfp':
        X = list(df['ecfp'])

    n = len(X)
    D = np.zeros([n, n])  # similarity matrix

    if dissimilarity is None:
        dissimilarity = lambda xi, xj: np.linalg.norm(xi - xj)
    elif dissimilarity == "Tanimoto":
        dissimilarity = lambda xi, xj: 1 - DataStructs.TanimotoSimilarity(xi, xj) 

    min_dsim = 0
    for i in range(n):
        for j in range(i, n):
            sim = dissimilarity(X[i], X[j])
            D[i, j] = sim
            D[j, i] = sim
            min_dsim = min(sim, min_dsim)
    D -= min_dsim

    # define k (cardinality constraints)
    if percentage is None:
        k = n
    else:
        k = int(n*percentage)

    # greedy
    def init_scores(rule):
        if rule == "maxsum":
            scores = np.zeros(n).reshape(n, 1)
        elif rule == "maxmin":
            scores = np.ones(n).reshape(n, 1) * np.inf
        return scores

    def update_scores(scores, j, rule):
        if rule == "maxsum":
            scores += D[:, j].reshape(n, 1)
        elif rule == "maxmin":
            scores = np.min(np.c_[scores, D[:, j].reshape(n, 1)], axis=1)
        return scores

    Z = list(range(n))
    rank = []
    while len(rank) < k:
        if len(rank) == 0:
            scores = init_scores(rule)
        remain_scores = scores[Z]
        if len(rank) == 0:
            j = Z[np.argmax(np.sum(D,axis=1))]
        else:
            j = Z[np.argmax(remain_scores)]
        scores = update_scores(scores, j, rule)
        rank.append(j)
        Z.remove(j)
    rank = np.append(rank, Z)
    df[rule+'_dissim_ranking'+vector] = np.argsort(rank)
    df.loc[df[rule+'_dissim_ranking'+vector] >= k, rule+'_dissim_ranking'+vector] = len(df)

    return df

def calculate_all_pairs_dissimilarity(fingerprints, similarity='Tanimoto', option='sum'):
    sum_sim = 0
    if similarity == 'Tanimoto':
        for fin in fingerprints:
            dissim_array = 1 - np.array(DataStructs.BulkTanimotoSimilarity(fin, fingerprints)) # fingerprints include fin
            if option == 'sum':
                sum_sim += np.sum(dissim_array)  
            elif option == 'min':
                sum_sim += np.sort(dissim_array)[1]
    n = len(fingerprints)
    if option == 'sum':
        ave_sim = sum_sim / (n * (n-1))
    elif option == 'min':
        ave_sim = sum_sim / n
    return ave_sim