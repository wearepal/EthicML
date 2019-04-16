#!/bin/bash

python -u run_experiments_tilde.py 0 > log/tilde_seed_0.txt 2>&1 &
python -u run_experiments_tilde.py 1 > log/tilde_seed_1.txt 2>&1 &
python -u run_experiments_tilde.py 2 > log/tilde_seed_2.txt 2>&1 &
python -u run_experiments_tilde.py 3 > log/tilde_seed_3.txt 2>&1 &
python -u run_experiments_tilde.py 4 > log/tilde_seed_4.txt 2>&1 &
python -u run_experiments_tilde.py 5 > log/tilde_seed_5.txt 2>&1 &
python -u run_experiments_tilde.py 6 > log/tilde_seed_6.txt 2>&1 &
python -u run_experiments_tilde.py 7 > log/tilde_seed_7.txt 2>&1 &
python -u run_experiments_tilde.py 8 > log/tilde_seed_8.txt 2>&1 &
python -u run_experiments_tilde.py 9 > log/tilde_seed_9.txt 2>&1 &
