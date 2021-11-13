#!/bin/sh

# out-of-domain
python out_of_domain.py --relu --seed 1 --task_targeted delaney+sampl+lipo --task_trained qm9 --model attentivefp --percentage 0.1

for i in 2 3 4 5
do
python out_of_domain.py --relu --use_saved_data --seed $i --task_targeted delaney+sampl+lipo --task_trained qm9 --model attentivefp --percentage 0.1
done

# out-of-domain normalize
python out_of_domain.py --normalize --relu --seed 1 --task_targeted delaney+sampl+lipo --task_trained qm9 --model attentivefp --percentage 0.1

for i in 2 3 4 5
do
python out_of_domain.py --normalize --relu --use_saved_data --seed $i --task_targeted delaney+sampl+lipo --task_trained qm9 --model attentivefp --percentage 0.1
done