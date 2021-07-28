# SUBMO
## Setup
```
$ conda install -y -c conda-forge rdkit
$ pip install -r requirements.txt
$ mkdir save_pickle
```
We have confirmed the behavior with the following versions.
```
conda: 4.8.4
python: 3.7.10
torch: 1.8.0
dgl-cu100: 0.5.3
dgllife: 0.2.6
rdkit: 2018.09.1
```

## Experiment
If you run the following script, you can train the GNN and select the diverse molecules.

```
$ ./experiment/run_experiment.sh
```

The results can be seen on `eval_wdud_mpd.ipynb`.