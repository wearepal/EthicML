#!/bin/bash

python -u run_experiments_beutel.py 0  > log/beutel_seed_0.txt 2>&1 &
python -u run_experiments_beutel.py 1  > log/beutel_seed_1.txt 2>&1 &
python -u run_experiments_beutel.py 2  > log/beutel_seed_2.txt 2>&1 &
python -u run_experiments_beutel.py 3  > log/beutel_seed_3.txt 2>&1 &
python -u run_experiments_beutel.py 4  > log/beutel_seed_4.txt 2>&1 &
python -u run_experiments_beutel.py 5  > log/beutel_seed_5.txt 2>&1 &
python -u run_experiments_beutel.py 6  > log/beutel_seed_6.txt 2>&1 &
python -u run_experiments_beutel.py 7  > log/beutel_seed_7.txt 2>&1 &
python -u run_experiments_beutel.py 8  > log/beutel_seed_8.txt 2>&1 &
python -u run_experiments_beutel.py 9  > log/beutel_seed_9.txt 2>&1 &
