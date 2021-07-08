from rdkit import rdBase
from rdkit.Chem import AllChem
import pickle
from utils.algorithms import greedy_baseline
import numpy as np
from utils.calc_chem import get_natom, get_nelec
print(rdBase.rdkitVersion)

# setting
tasks = ['qm9']
model_name = '_attentivefp_'
nums = ['_1', '_2', '_3', '_4', '_5']

# make df from pickle
df = {}
for task_name in tasks:
    if task_name not in df:
        df[task_name] = {}
        df[task_name + '_normalize'] = {}
    for num in nums:
        with open('./save_pickle/'+task_name+model_name+task_name+'_relu'+num+'_predicted_data.p', 'rb') as f:
            df[task_name][num] = pickle.load(f)
        with open('./save_pickle/'+task_name+model_name+task_name+'_normalize_relu'+num+'_predicted_data.p', 'rb') as f:
            df[task_name + '_normalize'][num] = pickle.load(f)


# calculate real values for qm9 properties
with open('./save_pickle/qm9_atom_edge_data.p', 'rb') as f:
    processed = pickle.load(f)

for task_name in df:
    for num in nums:
        for prop in df[task_name][num].columns:
            if prop == 'mols':
                break
            mean = processed[0][prop].mean()
            std = processed[0][prop].std()
            df[task_name][num]['pred ' + prop] = df[task_name][num]['pred ' + prop]*std + mean
            df[task_name][num][prop] = df[task_name][num][prop]*std + mean


# divide values by nelec or natom
for task_name in df:
    for num in nums:
        df[task_name][num]['Nelec'] = df[task_name][num]['mols'].apply(get_nelec)
        df[task_name][num]['Natom'] = df[task_name][num]['mols'].apply(get_natom)
        df[task_name][num]['u0/Nelec'] = df[task_name][num]['u0'] / df[task_name][num]['Nelec']
        df[task_name][num]['u298/Nelec'] = df[task_name][num]['u298'] / df[task_name][num]['Nelec']
        df[task_name][num]['h298/Nelec'] = df[task_name][num]['h298'] / df[task_name][num]['Nelec']
        df[task_name][num]['g298/Nelec'] = df[task_name][num]['g298'] / df[task_name][num]['Nelec']
        df[task_name][num]['cv/3n-6'] = df[task_name][num].apply(lambda x: x['cv'] / (3 * get_natom(x['mols']) - 6), axis = 1)


# calculate MACCS Key and ECFP
maccs = [AllChem.GetMACCSKeysFingerprint(mol) for mol in df['qm9']['_1']['mols']]
ecfp = [AllChem.GetMorganFingerprint(mol, 2) for mol in df['qm9']['_1']['mols']]

for task_name in df:
    for num in nums:
        df[task_name][num]['maccs'] = maccs
        df[task_name][num]['ecfp'] = ecfp

# selection using Tanimoto Coefficients
for task_name in df:
    for num in nums:
        df[task_name][num] = greedy_baseline(df[task_name][num], 0.01, 'Tanimoto', rule='maxsum', vector='_maccs')
        df[task_name][num] = greedy_baseline(df[task_name][num], 0.01, 'Tanimoto', rule='maxmin', vector='_maccs')
        df[task_name][num] = greedy_baseline(df[task_name][num], 0.01, 'Tanimoto', rule='maxsum', vector='_ecfp')
        df[task_name][num] = greedy_baseline(df[task_name][num], 0.01, 'Tanimoto', rule='maxmin', vector='_ecfp')


# add random ranking
n_mols = len(df['qm9']['_1'])
n_select = int(n_mols*0.01)
for num in nums:
    select = np.random.choice(range(n_mols), n_select, replace = False)
    random_rank = np.ones(n_mols) * n_mols
    random_rank[select] = np.arange(n_select)
    random_rank = list(map(int, random_rank))
    np.random.seed(int(num[-1]))
    for task_name in df:
        df[task_name][num]['random_ranking'] = random_rank

# save
pickle.dump(df, open( "./save_pickle/result_df.p", "wb" ))