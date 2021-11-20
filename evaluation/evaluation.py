from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import wasserstein_distance
import numpy as np
import pandas as pd
from rdkit import DataStructs


def calculate_prediction_scores(df, nums):
    ''' calculate average scores of MAE, MSE, RMSE, R2
    df: dataframe that have predicted scores
    nums: ex. ['_1', '_2', '_3']
    '''
    # calculate scores of each experiment
    score_dfs = {}
    for task_name in df:
        for prop in df[task_name]['_1'].columns:
            score_df = pd.DataFrame(index=['MAE', 'MSE', 'RMSE', 'R2'])
            if prop == 'mols':
                break
            for num in nums:
                y = df[task_name][num][prop].to_numpy()
                pred = df[task_name][num]['pred ' + prop].to_numpy()
                scores = [mean_absolute_error(y, pred),
                          mean_squared_error(y, pred),
                          np.sqrt(mean_squared_error(y, pred)),
                          r2_score(y, pred)]
                score_df[num] = scores
            score_dfs[task_name + ' ' + prop] = score_df

    # calculate average and standard diviation from score_dfs
    ave_std_df = pd.DataFrame(index=['MAE', 'MSE', 'RMSE', 'R2'])
    for task_prop in score_dfs:
        ave_std_df[task_prop+' average'] = score_dfs[task_prop].mean(
                                           axis='columns')
        ave_std_df[task_prop+' std'] = score_dfs[task_prop].std(
                                       axis='columns', ddof=0)
    return ave_std_df


def cdf_wdud(values, vmin, vmax):
    n = len(values)
    vrange = vmax - vmin
    values = np.append(values, vmin)
    values = np.append(values, vmax)
    values = np.sort(values)

    ans = 0
    for j in range(n+1):
        l, u = values[j], values[j+1]
        x = min(u, max(l, vmin + vrange * j / n))
        ans += (x - l) * (j / n + vmin / vrange - .5 * (x + l) / vrange) 
        ans += (u - x) * (- j / n - vmin / vrange + .5 * (u + x) / vrange) 
    return ans


def df2wd(df, methods, n_select, properties):
    # df is e.g., df['qm9']['_1']
    distance = pd.DataFrame([])    
    for prop in properties:
        prop_distance = pd.DataFrame([])
        prop_distance['method'] = methods
        prop_distance['property'] = prop
        distance_list = []
        vmin, vmax = min(df[prop]), max(df[prop])
        for method in methods:
            selection = df[df[method] < n_select][prop]
            distance_list.append(cdf_wdud(selection, vmin, vmax))   # evaluation.py cdf_wdud is used here
        prop_distance['distance'] = distance_list
        distance = pd.concat([distance, prop_distance])
    return distance


def dfs2wds(df, methods, n_select, properties):
    # df is e.g., df['qm9']
    distances = pd.DataFrame([])
    for num in df.keys():
        distance = df2wd(df[num], methods, n_select, properties)
        distance['num'] = num
        distances = pd.concat([distances, distance])
    return distances


def df2mpd(df, methods, n_select, task='None'):
    mpd = pd.DataFrame([])
    for fingerprint in ['maccs', 'ecfp']:
        fp_mpd = pd.DataFrame([])
        fp_mpd['method'] = methods
        fp_mpd['fingerprint'] = fingerprint
        if task != 'None':
            fp_mpd['fingerprint'] = task+'_'+fingerprint
        fp_mpd_list = []
        for method in methods:
            fps = list(df[df[method] < n_select][fingerprint])
            n = len(fps)
            sum_dissim = 0
            for fp in fps:
                dissim_array = 1 - np.array(DataStructs.BulkTanimotoSimilarity(fp, fps)) # fingerprints include fin
                sum_dissim += np.sum(dissim_array)              
            fp_mpd_list.append(sum_dissim / (n * (n-1)))
        fp_mpd['mpd'] = fp_mpd_list
        mpd = pd.concat([mpd, fp_mpd])
    return mpd


def dfs2mpd(df, methods, n_select, task='None'):
    # df is e.g., df['qm9']
    mpds = pd.DataFrame([])
    for num in df.keys():
        mpd = df2mpd(df[num], methods, n_select, task)
        mpd['num'] = num
        mpds = pd.concat([mpds, mpd])
    return mpds