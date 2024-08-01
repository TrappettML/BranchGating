#!/bin/bash
# param_loops.sh

model_names=('BranchModel')
branch_nums=(1 2 7) # 14 28 49 98 196 382 784)
# soma_funcs=('lse_0.1' 'lse_0.5' 'lse_1.0' 'lse_5.0' 'lse_10.0' 'softmax_0.1' 'softmax_1.0' 'softmax_2.0' 'softmaxsum_0.5' 'max' 'sum' 'median')
soma_funcs=('sum')
# sparsities=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
# repeats=(1 2 3 4 5)

for model in "${model_names[@]}"; do
    for repeat in "${repeats[@]}"; do
        for sparsity in "${sparsities[@]}"; do
            for branch_num in "${branch_nums[@]}"; do
                for soma_func in "${soma_funcs[@]}"; do
                    echo "--model_name $model --branch_num $branch_num --soma_func $soma_func --sparsity $sparsity --repeat $repeat"
                done
            done
        done
    done
done

