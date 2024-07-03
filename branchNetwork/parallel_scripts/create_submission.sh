#!/bin/bash

NUM_LINES=$(cat params.txt | wc -l)

sbatch --array=71-$NUM_LINES futures_sbatch.sh

