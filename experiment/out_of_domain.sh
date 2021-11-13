#!/bin/sh

# out-of-domain
python test_embeddings.py --relu --seed 1 --task_targeted delaney+sampl+lipo --task_trained qm9 --model attentivefp --percentage 0.1

for i in 2 3 4 5
do
python test_embeddings.py --relu --use_saved_data --seed $i --task_targeted delaney+sampl+lipo --task_trained qm9 --model attentivefp --percentage 0.1
done

# out-of-domain normalize
python test_embeddings.py --normalize --relu --seed 1 --task_targeted delaney+sampl+lipo --task_trained qm9 --model attentivefp --percentage 0.1

for i in 2 3 4 5
do
python test_embeddings.py --normalize --relu --use_saved_data --seed $i --task_targeted delaney+sampl+lipo --task_trained qm9 --model attentivefp --percentage 0.1
done