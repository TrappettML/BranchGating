#!/bin/bash
# param_loops.sh

model_names=('BranchModel')
branch_nums=(1 2 7 14 28 49 98 196 382 784)
soma_funcs=('softmax_0.1' 'softmax_1.0' 'softmax_2.0' 'softmaxsum_0.1' 'softmaxsum_0.5' 'softmaxsum_1.0' 'softmaxsum_2.0' 'max' 'sum' 'median')
sparsities=(0.0)
repeats=(1)

for model in "${model_names[@]}"; do
    for branch_num in "${branch_nums[@]}"; do
        for soma_func in "${soma_funcs[@]}"; do
            for sparsity in "${sparsities[@]}"; do
                for repeat in "${repeats[@]}"; do
                    echo "--model_name $model --branch_num $branch_num --soma_func $soma_func --sparsity $sparsity --repeat $repeat"
                done
            done
        done
    done
done

