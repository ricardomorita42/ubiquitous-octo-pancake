#!/bin/bash
for i in {1..55} ; do
	for j in {1..5} ; do
		sed -i '5 i \\t\"feature\": \"MFCC\",' experiments/configs/exp-$i.$j.json
	done
done
