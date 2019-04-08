#!/bin/bash

python -u run_experiments.py 0 > log/x_seed_0.txt 2>&1 &
python -u run_experiments.py 1 > log/x_seed_1.txt 2>&1 &
python -u run_experiments.py 2 > log/x_seed_2.txt 2>&1 &
python -u run_experiments.py 3 > log/x_seed_3.txt 2>&1 &
python -u run_experiments.py 4 > log/x_seed_4.txt 2>&1 &
python -u run_experiments.py 5 > log/x_seed_5.txt 2>&1 &
python -u run_experiments.py 6 > log/x_seed_6.txt 2>&1 &
python -u run_experiments.py 7 > log/x_seed_7.txt 2>&1 &
python -u run_experiments.py 8 > log/x_seed_8.txt 2>&1 &
python -u run_experiments.py 9 > log/x_seed_9.txt 2>&1 &
