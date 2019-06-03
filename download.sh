#!/usr/bin/bash

base="https://people.sc.fsu.edu/~jburkardt/datasets/tsp/"
files=( "att48.tsp" "att48_d.txt" "att48_xy.txt" "att48_s.txt")

for file in "${files[@]}"
do
	wget $base$file
done
