#!/bin/bash
for i in {31..60} ; do
	for j in {1..5} ; do
		python3 test.py -c experiments/configs/exp-$i.$j.json
		echo "python3 test.py -c experiments/configs/exp-$i.$j.json" >> run_status.txt
	done
done
