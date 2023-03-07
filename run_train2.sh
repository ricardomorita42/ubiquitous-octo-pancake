#!/bin/bash
for i in {56..56} ; do
	for j in {1..5} ; do
		python3 train.py -c experiments/configs/exp-$i.$j.json
		echo "python3 train.py -c experiments/configs/exp-$i.$j.json" >> run_status.txt
	done
done
