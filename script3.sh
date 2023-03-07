#!/bin/bash
python3 train.py -c experiments/configs/exp-29.1.json | tee tests/test-29.1.txt
python3 train.py -c experiments/configs/exp-29.2.json | tee tests/test-29.2.txt
python3 train.py -c experiments/configs/exp-29.3.json | tee tests/test-29.3.txt
python3 train.py -c experiments/configs/exp-29.4.json | tee tests/test-29.4.txt
python3 train.py -c experiments/configs/exp-29.5.json | tee tests/test-29.5.txt

python3 train.py -c experiments/configs/exp-30.1.json | tee tests/test-30.1.txt
python3 train.py -c experiments/configs/exp-30.2.json | tee tests/test-30.2.txt
python3 train.py -c experiments/configs/exp-30.3.json | tee tests/test-30.3.txt
python3 train.py -c experiments/configs/exp-30.4.json | tee tests/test-30.4.txt
python3 train.py -c experiments/configs/exp-30.5.json | tee tests/test-30.5.txt

python3 train.py -c experiments/configs/exp-31.1.json | tee tests/test-31.1.txt
python3 train.py -c experiments/configs/exp-31.2.json | tee tests/test-31.2.txt
python3 train.py -c experiments/configs/exp-31.3.json | tee tests/test-31.3.txt
python3 train.py -c experiments/configs/exp-31.4.json | tee tests/test-31.4.txt
python3 train.py -c experiments/configs/exp-31.5.json | tee tests/test-31.5.txt

python3 train.py -c experiments/configs/exp-32.1.json | tee tests/test-32.1.txt
python3 train.py -c experiments/configs/exp-32.2.json | tee tests/test-32.2.txt
python3 train.py -c experiments/configs/exp-32.3.json | tee tests/test-32.3.txt
python3 train.py -c experiments/configs/exp-32.4.json | tee tests/test-32.4.txt
python3 train.py -c experiments/configs/exp-32.5.json | tee tests/test-32.5.txt

python3 train.py -c experiments/configs/exp-33.1.json | tee tests/test-33.1.txt
python3 train.py -c experiments/configs/exp-33.2.json | tee tests/test-33.2.txt
python3 train.py -c experiments/configs/exp-33.3.json | tee tests/test-33.3.txt
python3 train.py -c experiments/configs/exp-33.4.json | tee tests/test-33.4.txt
python3 train.py -c experiments/configs/exp-33.5.json | tee tests/test-33.5.txt

