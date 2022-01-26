# SUBMO
![overview](https://user-images.githubusercontent.com/45445358/151129384-4db3bc73-6cb6-4ac0-a236-f7bf9deb8ab7.png)

## Setup
The required modules can be obtained by the following commands.
For torch and dgl installation, change the contents of the `requirements.txt` file according to your environment.
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

The results can be seen by running the cells in `eval_wdud_mpd.ipynb`.
