#!/bin/sh

# qm9 relu
python run_experiment.py --relu --seed 1 --save_model --tasks qm9 --epoch 300 --dim 280 --dropout 0.5 --lr 0.0004 --batch_size 500 --percentage 0.015

for i in 2 3 4 5
do
python run_experiment.py --relu --use_saved_data --seed $i --save_model --tasks qm9 --model attentivefp --epoch 300 --dim 280 --dropout 0.5 --lr 0.0004 --batch_size 500 --percentage 0.015
done

# qm9 relu normalize
python run_experiment.py --normalize --relu --seed 1 --save_model --tasks qm9 --epoch 300 --dim 280 --dropout 0.5 --lr 0.0004 --batch_size 500 --percentage 0.015

for i in 2 3 4 5
do
python run_experiment.py --normalize --relu --use_saved_data --seed $i --save_model --tasks qm9 --model attentivefp --epoch 300 --dim 280 --dropout 0.5 --lr 0.0004 --batch_size 500 --percentage 0.015
done

# run df update
python write_df_pickle.py