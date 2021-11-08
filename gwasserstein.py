
from utils.algorithms import greedy_wasserstein
import pickle
df = pickle.load(open("./save_pickle/result_df.p", "rb"))
df['qm9']['_1'] = greedy_wasserstein(df['qm9']['_1'], 0.01, is_pred=False)
df['qm9']['_2'] = greedy_wasserstein(df['qm9']['_2'], 0.01, is_pred=True)
with open('save_pickle/resulet_df_wasserstein_predFalse.p', 'wb') as f:
    pickle.dump(df, f)