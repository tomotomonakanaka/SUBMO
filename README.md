# SubMo-GNN
This repository is the implementation of [Selecting molecules with diverse structures and properties by maximizing submodular functions of descriptors learned with graph neural networks](https://www.nature.com/articles/s41598-022-04967-9).
![image (3)](https://user-images.githubusercontent.com/45445358/151144208-13d07213-0215-4418-9de9-fe2bfbf90479.png)
![overview](https://user-images.githubusercontent.com/45445358/151129384-4db3bc73-6cb6-4ac0-a236-f7bf9deb8ab7.png)

## Setup
The required modules can be obtained by the following commands.
For torch and dgl installation, change `requirements.txt` according to your environment.
```
$ conda install -y -c conda-forge rdkit
$ pip install -r requirements.txt
$ mkdir save_pickle
```
We have confirmed the implementation with the following versions.
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

## License
The source code is licensed MIT.
