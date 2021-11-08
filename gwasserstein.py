
from utils.algorithms import greedy_wasserstein
import pickle
df = pickle.load(open("./save_pickle/result_df.p", "rb"))
nums = ['_1', '_2', '_3', '_4', '_5']
for num in nums:
    df['qm9'][num] = greedy_wasserstein(df['qm9'][num], 0.01, is_pred=False)
    df['qm9'][num] = greedy_wasserstein(df['qm9'][num], 0.01, is_pred=True)
with open('save_pickle/resulet_df_wasser.p', 'wb') as f:
    pickle.dump(df, f)