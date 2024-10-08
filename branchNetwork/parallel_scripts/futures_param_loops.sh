#!/bin/bash
# param_loops.sh

model_names=('BranchModel')
branch_nums=(1 2 7 14 28 49 98 196 382 784)
# branch_nums=(1 2 5 10 20 50 100 200 400 800 1200)
# soma_funcs=('tanh' 'sigmoid' 'softplus' 'softsign' 'elu' 'gelu' 'selu')
soma_funcs=('sum')
sparsities=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0) #(0.0 0.5) # 
repeats=(1 2 3 4 5)
hiddens=(784) # (50 100 200 400)
n_npbs=(1) # 5 10 20 50 100 200 400 800 1200)
lrules=('rl')
det_gates=(0 1)

for model in "${model_names[@]}"; do
    for repeat in "${repeats[@]}"; do
        for branch_num in "${branch_nums[@]}"; do
            for soma_func in "${soma_funcs[@]}"; do
                for hidden in "${hiddens[@]}"; do
                    for sparsity in "${sparsities[@]}"; do
                        for lrule in "${lrules[@]}"; do
                            for det_gate in "${det_gates[@]}"; do
                                echo "--model_name $model --n_branches $branch_num --soma_func $soma_func --sparsity $sparsity --repeat $repeat --hidden [$hidden,$hidden] --learning_rule ${lrule} --determ_gates $det_gate"
                            done
                        done
                    done
                done
            done
        done
    done
done

