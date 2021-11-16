from rdkit import rdBase
from rdkit.Chem import AllChem
import pickle
from utils.algorithms import greedy_baseline, greedy_wasserstein
import numpy as np
from utils.calc_chem import get_natom, get_nelec
import time
print(rdBase.rdkitVersion)

# setting
tasks = ['delaney']
model_name = '_attentivefp_'
nums = ['_1', '_2', '_3', '_4', '_5']
props = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']

# make df from pickle
df = {}
for task_name in tasks:
    if task_name not in df:
        df[task_name] = {}
        df[task_name + '_normalize'] = {}
    for num in nums:
        with open('./save_pickle/'+task_name+model_name+'qm9'+'_relu'+num+'_data.pickle', 'rb') as f:
            df[task_name][num] = pickle.load(f)
        with open('./save_pickle/'+task_name+model_name+'qm9'+'_normalize_relu'+num+'_data.pickle', 'rb') as f:
            df[task_name + '_normalize'][num] = pickle.load(f)


# calculate real values for qm9 properties
with open('./save_pickle/raw_train_df.p', 'rb') as f:
    raw_train_df = pickle.load(f)

for task_name in df:
    for num in nums:
        for prop in props:
            mean = raw_train_df[prop].mean()
            std = raw_train_df[prop].std()
            df[task_name][num]['pred ' + prop] = df[task_name][num]['pred ' + prop]*std + mean


# calculate MACCS Key and ECFP
for task_name in df:
    maccs = [AllChem.GetMACCSKeysFingerprint(mol) for mol in df[task_name]['_1']['mols']]
    ecfp = [AllChem.GetMorganFingerprint(mol, 2) for mol in df[task_name]['_1']['mols']]
    for num in nums:
        df[task_name][num]['maccs'] = maccs
        df[task_name][num]['ecfp'] = ecfp

# selection using Tanimoto Coefficients
for task_name in df:
    time_MSMK = time.time()
    df[task_name]['_1'] = greedy_baseline(df[task_name]['_1'], 0.1, 'Tanimoto', rule='maxsum', vector='_maccs')
    time_MMMK = time.time()
    df[task_name]['_1'] = greedy_baseline(df[task_name]['_1'], 0.1, 'Tanimoto', rule='maxmin', vector='_maccs')
    time_MSEF = time.time()
    df[task_name]['_1'] = greedy_baseline(df[task_name]['_1'], 0.1, 'Tanimoto', rule='maxsum', vector='_ecfp')
    time_MMEF = time.time()
    df[task_name]['_1'] = greedy_baseline(df[task_name]['_1'], 0.1, 'Tanimoto', rule='maxmin', vector='_ecfp')
    time_bs = time.time()
    print('MSMK', time_MMMK-time_MSMK)
    print('MMMK', time_MSEF-time_MMMK)
    print('MSEF', time_MMEF-time_MSEF)
    print('MMEF', time_bs-time_MMEF)

for task_name in df:
    for num in nums:
        time_WS = time.time()
        df[task_name][num] = greedy_wasserstein(df[task_name][num], 0.1)
        time_end = time.time()
        print('Wasser', time_end-time_WS)


# add random ranking
for task_name in df:
    n_mols = len(df[task_name]['_1'])
    n_select = int(n_mols*0.1)
    for num in nums:
        select = np.random.choice(range(n_mols), n_select, replace = False)
        random_rank = np.ones(n_mols) * n_mols
        random_rank[select] = np.arange(n_select)
        random_rank = list(map(int, random_rank))
        np.random.seed(int(num[-1]))
        for task_name in df:
            df[task_name][num]['random_ranking'] = random_rank

# save
pickle.dump(df, open( "./save_pickle/result_df_delaney.p", "wb" ))