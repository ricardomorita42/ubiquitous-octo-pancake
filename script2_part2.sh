#!/bin/bash
python3 train.py -c experiments/configs/exp-39.1.json | tee tests/test-39.1.txt
python3 train.py -c experiments/configs/exp-39.2.json | tee tests/test-39.2.txt
python3 train.py -c experiments/configs/exp-39.3.json | tee tests/test-39.3.txt
python3 train.py -c experiments/configs/exp-39.4.json | tee tests/test-39.4.txt
python3 train.py -c experiments/configs/exp-39.5.json | tee tests/test-39.5.txt

python3 train.py -c experiments/configs/exp-40.1.json | tee tests/test-40.1.txt
python3 train.py -c experiments/configs/exp-40.2.json | tee tests/test-40.2.txt
python3 train.py -c experiments/configs/exp-40.3.json | tee tests/test-40.3.txt
python3 train.py -c experiments/configs/exp-40.4.json | tee tests/test-40.4.txt
python3 train.py -c experiments/configs/exp-40.5.json | tee tests/test-40.5.txt

python3 train.py -c experiments/configs/exp-41.1.json | tee tests/test-41.1.txt
python3 train.py -c experiments/configs/exp-41.2.json | tee tests/test-41.2.txt
python3 train.py -c experiments/configs/exp-41.3.json | tee tests/test-41.3.txt
python3 train.py -c experiments/configs/exp-41.4.json | tee tests/test-41.4.txt
python3 train.py -c experiments/configs/exp-41.5.json | tee tests/test-41.5.txt

python3 train.py -c experiments/configs/exp-42.1.json | tee tests/test-42.1.txt
python3 train.py -c experiments/configs/exp-42.2.json | tee tests/test-42.2.txt
python3 train.py -c experiments/configs/exp-42.3.json | tee tests/test-42.3.txt
python3 train.py -c experiments/configs/exp-42.4.json | tee tests/test-42.4.txt
python3 train.py -c experiments/configs/exp-42.5.json | tee tests/test-42.5.txt

python3 train.py -c experiments/configs/exp-43.1.json | tee tests/test-43.1.txt
python3 train.py -c experiments/configs/exp-43.2.json | tee tests/test-43.2.txt
python3 train.py -c experiments/configs/exp-43.3.json | tee tests/test-43.3.txt
python3 train.py -c experiments/configs/exp-43.4.json | tee tests/test-43.4.txt
python3 train.py -c experiments/configs/exp-43.5.json | tee tests/test-43.5.txt

