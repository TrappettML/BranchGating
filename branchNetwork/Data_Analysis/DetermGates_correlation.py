import torch
import numpy as np
from scipy.stats import sem, iqr
from numpy import median

from collections import defaultdict
from pprint import pprint

import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from matplotlib.colors import Normalize

import pickle
import os
import re

from sklearn.metrics.pairwise import cosine_similarity
import ray
from tqdm import tqdm
from ipdb import set_trace
import time

from branchNetwork.architectures.BranchMM import BranchModel
from branchNetwork.dataloader import load_rotated_flattened_mnist_data
from branchNetwork.Data_Analysis.ft_rem_pipeline import make_plots_folder
from branchNetwork.architectures.reinforce_criterion import RLCrit



os.environ['OMP_NUM_THREADS'] = '2'
torch.set_num_threads(2)

for i in tqdm(range(10), disable=True):
    time.sleep(1)

def parse_filename(filename: str, file_type='state_dict') -> tuple:
    # Regex patterns for different filename types
    # pattern1 = r"si_all_task_accuracies_learn_gates_False_soma_func_sum_l2_0.0_lr_0.0001_n_npb_([^_]*)_([^_]*)_n_branches_([^_]*)_([^_]*)_sparsity_([^_]*)_hidden_784_784_det_masks_True_model_name_BranchModel_repeat_([^_]*)_epochs_per_task_20.pkl"
    # pattern2 = r"si_sequential_eval_task_metrics_learn_gates_False_soma_func_sum_l2_0.0_lr_0.0001_n_npb_([^_]*)_([^_]*)_n_branches_([^_]*)_([^_]*)_sparsity_([^_]*)_hidden_784_784_det_masks_True_model_name_BranchModel_repeat_([^_]*)_epochs_per_task_20.pkl"
    state_dict_pattern = 'state_dict__learn_gates_False_soma_func_sum_l2_0.0_lr_0.0001_n_npb_([^_]*)_([^_]*)_n_branches_([^_]*)_([^_]*)_sparsity_([^_]*)_hidden_784_784_det_masks_([^_]*)_model_name_BranchModel_repeat_([^_]*)_epochs_per_task_20'
    config_pattern = 'configs_learn_gates_False_soma_func_sum_l2_0.0_lr_0.0001_n_npb_([^_]*)_([^_]*)_n_branches_([^_]*)_([^_]*)_sparsity_([^_]*)_hidden_784_784_det_masks_([^_]*)_model_name_BranchModel_repeat_([^_]*)_epochs_per_task_20.pkl'
    
    # Try matching with the first pattern
    if file_type == 'state_dict':
        matches = re.search(state_dict_pattern, filename)
    elif file_type == 'config':
        matches = re.search(config_pattern, filename)
    
    if matches:
        # Extract values and convert to appropriate types
        sparsity = float(matches.group(5))
        n_branch_0, n_branch_1 =int(matches.group(3)), int(matches.group(4))
        n_npb_0, n_npb_1 = int(matches.group(1)), int(matches.group(2))
        determ = str(matches.group(6))
        repeat = int(matches.group(7))
        return (sparsity, n_branch_0, determ, repeat)
    else:
        return None
    
def rl_filename_parser(filename: str, file_type='state_dict') -> tuple:
    state_pattern = r"state_dict__soma_func_sum_lr_0.0001_loss_func_RLCrit_n_npb_([^_]*)_([^_]*)_n_branches_([^_]*)_([^_]*)_sparsity_([^_]*)_det_masks_([^_]*)_model_name_BranchModel_repeat_([^_]*)_epochs_per_task_20"
    config_pattern = r"configs_soma_func_sum_lr_0.0001_loss_func_RLCrit_n_npb_([^_]*)_([^_]*)_n_branches_([^_]*)_([^_]*)_sparsity_([^_]*)_det_masks_([^_]*)_model_name_BranchModel_repeat_([^_]*)_epochs_per_task_20.pkl"
    if file_type == 'state_dict':
        pattern = state_pattern
    elif file_type == 'config':
        pattern = config_pattern
    matches = re.search(pattern, filename)
    sparsity = float(matches.group(5))
    n_branch = int(matches.group(3))
    n_npb = int(matches.group(1))
    repeat = int(matches.group(7))
    det_mask = str(matches.group(6))
    # set_trace()
    return (sparsity, n_branch, det_mask, repeat)

def load_state_models(directory: str, file_parser=parse_filename) -> tuple:
    # Dictionaries to hold data from all pickle files
    state_dicts = {}
    config_paths = {}
    # Loop through each file in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # print(file)
            if 'state_dict' in file:
                # print(file)
                # Construct the full path to the file
                file_path = os.path.join(root, file)
                # Parse the filename into a tuple of variables and determine the data type
                state_dict_key = file_parser(file, 'state_dict')
                if state_dict_key:
                    state_dicts[state_dict_key] = file_path   
            elif 'configs' in file:
                file_path = os.path.join(root, file)
                config_key = file_parser(file, 'config')
                config_paths[config_key] = file_path
    return state_dicts, config_paths

def get_model(config_file: str, state_file: str) -> tuple:
    with open(config_file, 'rb') as f:
        config = pickle.load(f)
    configs = 'dict(' + config + ')'
    configs = eval(configs)
    for key, value in {'n_out':10, 'n_in':784, 'n_contexts':len(configs['TRAIN_CONFIGS']['rotation_degrees']), 'device':'cpu', 'dropout':0}.items():
        configs['MODEL_CONFIGS'][key] = value
    model = BranchModel(configs['MODEL_CONFIGS'])
    model.load_state_dict(torch.load(state_file, weights_only=False))
    model.eval()
    rotations = configs['TRAIN_CONFIGS']['rotation_degrees']
    return model, rotations

def get_activations(model, test_loader, rotation, labels=[5], rep_len=200):
    # Dictionary to hold activations, separated by label and layer
    activations = defaultdict(lambda: defaultdict(list))
    layer_names = ['b1', 'b2','s1', 's2', 'h1', 'h2']  # Adjust as per your model
    label_count = 0
    # rep_len = 20
    # print(f'_______________{label_counts=}__________________')
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            
            # set_trace()
            if label_count > rep_len:
                break
            _ = model(x, rotation)
            branch1 = model.branch_activities_1
            branch2 = model.branch_activities_2
            soma1 = model.soma_activities_1
            soma2 = model.soma_activities_2
            hidden1 = model.x1_hidden
            hidden2 = model.x2_hidden

            # Group activations by layer for easier access
            all_activations = [branch1, branch2, soma1, soma2, hidden1, hidden2]
            
            # for label in labels:
            for label in y.unique():
                if label in labels:
                    mask = (y == label)
                    # set_trace()
                    label_count += mask.sum().item()
                    for layer_name, layer_activations in zip(layer_names, all_activations):
                        # Append filtered activations for the current label
                        filtered_activations = layer_activations[mask]
                        activations[layer_name][label.item()].append(filtered_activations)
                        
                        # label_counts[label.item()] += mask.sum().item()
                        # print(f'{label_counts=}')
            
    return {str(rotation):activations}

def reshape_activations(activations_dict, rep_len=200):
    # set_trace()
    altered_act_dict = defaultdict(dict)
    for rot, act_dict in activations_dict.items():
        b1_activations = act_dict['b1']
        b2_activations = act_dict['b2']
        s1_activations = act_dict['s1']
        s2_activations = act_dict['s2']
        hidden1_activations = act_dict['h1']
        hidden2_activations = act_dict['h2']
        for key in b1_activations.keys():
            size_b, size_br, size_n = b1_activations[key][0].shape
            # print(f'{key=}')
            b1_activations[key] = torch.cat(b1_activations[key], dim=0)
            b2_activations[key] = torch.cat(b2_activations[key], dim=0)
            s1_activations[key] = torch.cat(s1_activations[key], dim=0)[:rep_len, :]
            s2_activations[key] = torch.cat(s2_activations[key], dim=0)[:rep_len, :]
            hidden1_activations[key] = torch.cat(hidden1_activations[key], dim=0)[:rep_len, :]
            hidden2_activations[key] = torch.cat(hidden2_activations[key], dim=0)[:rep_len, :]
            b1_activations[key] = b1_activations[key].transpose(0, 1).reshape(-1, size_br*784)[:rep_len, :]
            b2_activations[key] = b2_activations[key].transpose(0, 1).reshape(-1, size_br*784)[:rep_len, :]
        altered_act_dict[rot] = {'b1':b1_activations, 'b2':b2_activations,'s1':s1_activations, 's2':s2_activations, 'h1':hidden1_activations, 'h2':hidden2_activations}
    return altered_act_dict

def custom_correlation_matrix(tensor):
    # Convert the tensor to a NumPy array
    tensor_np = tensor.detach()
    # Compute mean and standard deviation
    mean = torch.mean(tensor_np, axis=0)
    stddev = torch.std(tensor_np, axis=0) + 1e-10 # Avoid division by zero
    
    # Normalize the tensor data
    tensor_normalized = (tensor_np - mean) / (stddev )  

    # Calculate the correlation matrix using dot product
    tensor_size = tensor_normalized.size(0)
    correlation_matrix = torch.mm(tensor_normalized.T, tensor_normalized) / tensor_size

    # Set correlations to zero where either of the features has zero standard deviation
    zero_var_indices = (stddev <= 1e-10).nonzero(as_tuple=True)[0]
    correlation_matrix.index_fill_(1, zero_var_indices, 0)
    correlation_matrix.index_fill_(0, zero_var_indices, 0)

    return correlation_matrix.numpy()

def get_correlation_for_degree_label(correlations_by_layer_label):
    """each value in the reshaped_activations_dict.value is a dict containing activations for each label.
    We want the similarity of all the activations organized by label.
    args: correlations_by_layer_label: dict[angle][layer][label] = activations(batch_size, n_activitiies)
    returns: dict[angel][layer][label] = similarity_score)"""
    results = {}
    for rot, act_dicts in correlations_by_layer_label.items():
        layer_correlations = defaultdict(dict)
        for layer_label, class_activations in act_dicts.items():
            for label, activations in class_activations.items():
                # print(f'{activations.shape=}')
                layer_correlations[layer_label][str(label)] = custom_correlation_matrix(activations)
        results[str(rot)] = layer_correlations
    return results

def get_similarity_between_two_tasks_same_label(correlation_1, correlation_2):
    """Get the similarity betwee two correlation matrices"""
    c_1 = correlation_1.flatten()
    c_2 = correlation_2.flatten()
    # set_trace()
    return cosine_similarity([c_1], [c_2])

def similarity_between_all_tasks(all_task_all_labels_correlations, rotation_degrees):
    """Get the similarity between all tasks with the same label
    For each Label, we need to loop through all tasks and compare the similar label for each layer activation. 
    args: all_task_all_labels_correlations: dict[task][layer][label] = correlation_matrix
            rotation_degrees: list of rotation degrees/tasks
            
    returns: dict[(task2,  layer, label)] = similarity_score
    """
    Layers = ['b1', 'b2','s1', 's2', 'h1', 'h2']
    task1 = str(rotation_degrees[0])
    similarities = {}
    for task2 in rotation_degrees:
        task2 = str(task2)
        for layer in Layers:
            labels = list(all_task_all_labels_correlations[task1][layer].keys()) # labels are determined at activation acquisition time
            for label in labels:
                similarities[(task2, layer, label)] = get_similarity_between_two_tasks_same_label(all_task_all_labels_correlations[task1][layer][label], all_task_all_labels_correlations[task2][layer][label])
    return similarities

def gen_correlations_for_degree_pipeline(model, rotation):
    test_loader, _ = load_rotated_flattened_mnist_data(batch_size=200, rotation_in_degrees=rotation)
    activations_dict = get_activations(model, test_loader, rotation)
    activations_dict_reshaped = reshape_activations(activations_dict)
    return get_correlation_for_degree_label(activations_dict_reshaped)

def representation_pipeline(model_key: tuple, state_paths: dict, config_paths: dict, rotation: int) -> dict:
    model, _ = get_model(config_paths[model_key], state_paths[model_key])
    test_loader, _ = load_rotated_flattened_mnist_data(batch_size=200, rotation_in_degrees=rotation)
    activations_dict = get_activations(model, test_loader, rotation)
    activations_dict_reshaped = reshape_activations(activations_dict)
    return activations_dict_reshaped

def corr_similarity_pipeline(model_key, state_paths, config_paths):
    model, rotations = get_model(config_paths[model_key], state_paths[model_key])
    all_correlations = [gen_correlations_for_degree_pipeline(model, rotation) for rotation in tqdm(rotations, desc=f'Rotation: {model_key}')]
    all_correlations_dict = {k:v for d in all_correlations for k, v in d.items()}
    del all_correlations
    return {model_key: similarity_between_all_tasks(all_correlations_dict, rotations)}

def custom_mean(t):
    tensor_np = t.detach()
    # Compute mean and standard deviation
    mean = torch.mean(tensor_np, axis=0)
    return mean
    
def mean_similarity_pipeline(model_key, state_paths, config_paths):
    model, rotations = get_model(config_paths[model_key], state_paths[model_key])
    all_means = []
    for rotation in rotations:
        test_loader, _ = load_rotated_flattened_mnist_data(batch_size=200, rotation_in_degrees=rotation)
        activations_dict = get_activations(model, test_loader, rotation)
        activations_dict_reshaped = reshape_activations(activations_dict)
        results = {}
        for rot, act_dicts in activations_dict_reshaped.items():
            layer_means = defaultdict(dict)
            for layer_label, class_activations in act_dicts.items():
                for label, activations in class_activations.items():
                    # print(f'{activations.shape=}')
                    layer_means[layer_label][str(label)] = custom_mean(activations)
            results[str(rot)] = layer_means
            all_means.append(results)
    all_means = {k:v for d in all_means for k, v in d.items()}
    return {model_key: similarity_between_all_tasks(all_means, rotations)}

    
def plot_all_activations(data_dict, file_name='activations_plot.png'):
    # Extract continuous values and use them for the color map
    continuous_values = sorted(data_dict.keys())
    norm = Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=viridis, norm=norm)

    # Define the subplot grid and figure size
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    layer_names = ['b1','b2','s1','s2', 'h1','h2']  # Order of layers to plot

    # Loop through each layer and plot in the designated subplot
    for i, layer in enumerate(layer_names):
        ax = axs[i // 2, i % 2]

        for _k in continuous_values:
            # Extract data for the specified layer for each m value
            layer_data = data_dict[_k]
            layer_data = {k: v for k, v in layer_data.items() if k[1] == layer}
            
            # Prepare data for plotting
            angles = [int(k[0]) for k in layer_data.keys()]  # Rotation angles
            values = [v.item() for v in layer_data.values()]  # Activation values
            
            # Sorting the data by angle for better visualization
            sorted_indices = np.argsort(angles)
            angles = np.array(angles)[sorted_indices]
            values = np.array(values)[sorted_indices]

            # Plotting
            color = sm.to_rgba(norm(_k[0]))
            ax.plot(angles, values, marker='o', linestyle='-', color=color)
            ax.set_xlabel('Rotation Angle')
            ax.set_ylabel('Cosine Similarity Score')

        ax.set_title(f'Layer {layer}')
        ax.grid(True)
        ax.set_xticks(angles)  # Ensure x-ticks show actual angles

    # Create color bar in the space made by fig.subplots_adjust
    cbar_ax = fig.add_axes([1.02, 0.25, 0.01, 0.5]) # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Sparsity')

    # Adjust layout and add a supertitle
    fig.suptitle(f'Similarity of 0° compared to other Rotations with n_b:{_k[1]}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'./{file_name}', bbox_inches='tight')

# @ray.remote  
def full_similarity_pipeline(n_branches, state_paths, config_paths, sparsities, repeat=1):
    # r_corr_similarity_pipeline = ray.remote(corr_similarity_pipeline)
    # set_trace()
    target_keys = [(sparsity, n_branches, int(784/n_branches), repeat) for sparsity in sparsities]
    all_similarities = [corr_similarity_pipeline(key, state_paths, config_paths) for key in tqdm(target_keys, desc=f'branches: {n_branches}')]
    all_similarities_dict = {k:v for d in all_similarities for k,v in d.items()}
    plot_all_activations(all_similarities_dict, f'branchNetwork/Data_Analysis/activations_plot_n_b_{n_branches}.png')
    

def plot_all_activations_true_false(data_dict1, data_dict2, metric='mean', data_dict_labels=['Determ True', 'Determ False'], file_name='activations_plot.png'):
    # Extract continuous values and use them for the color map
    # continuous_values = sorted(data_dict1.keys())
    norm = Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=viridis, norm=norm)

    # Define the subplot grid and figure size
    fig, axs = plt.subplots(6, 2, figsize=(15, 17), sharey=True, layout='constrained')
    layer_names = ['b1','s1', 'h1', 'b2', 's2', 'h2']  # Order of layers to plot

    # Loop through each layer and plot in the designated subplot
    for col, data_dict in enumerate([data_dict1, data_dict2]):
        for i, layer in enumerate(layer_names):
            ax = axs[i, col]
            seen_combinations = set()
            
            continuous_values = sorted(data_dict.keys())
            for key in continuous_values:
                sparsity, n_b, det_bool, repeat = key
                if (sparsity, n_b, det_bool) not in seen_combinations:
                    seen_combinations.add((sparsity, n_b, det_bool))
                    all_repeats = []
                    all_repeats = []
                    # Collect data across all repeats for this combination
                    for repeat_key in [k for k in data_dict.keys() if k[:3] == (sparsity, n_b, det_bool)]:
                        layer_data = data_dict[repeat_key]
                        layer_specific_data = {k: v for k, v in layer_data.items() if k[1] == layer}
                        angles = [int(k[0]) for k in layer_specific_data.keys()]
                        values = np.array([v.item() for v in layer_specific_data.values()])
                        sorted_indices = np.argsort(angles)
                        angles = np.array(angles)[sorted_indices]
                        values = np.array(values)[sorted_indices]
                        all_repeats.append(values)

                        
                if metric == 'mean':
                    plot_values = np.mean(all_repeats, axis=0)
                    error_bars = sem(all_repeats, axis=0)
                elif metric == 'median':
                    plot_values = median(all_repeats, axis=0)
                    error_bars = iqr(all_repeats, axis=0)
                    
                # ax.plot(angles, )
                # Plotting
                # print(f'{_r[0]}')
                color = sm.to_rgba(norm(sparsity))
                # ax.plot(angles, plot_values, marker='o', linestyle='-', color=color)
                # print(f'{error_bars}')
                ax.errorbar(angles, plot_values, yerr=error_bars, fmt='o-', color=color
                            , label=f'{data_dict_labels[col]}')
              
                ax.set_ylabel(f'Layer: {layer}')
            ax.set_title(f'Gating type: {data_dict_labels[col]}', fontsize='small', loc='left')
            ax.grid(True)
            ax.set_xticks(angles)  # Ensure x-ticks show actual angles

    # Create color bar in the space made by fig.subplots_adjust
    cbar_ax = fig.add_axes([1.02, 0.25, 0.01, 0.5]) # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Sparsity')
    fig.supylabel('Cosine Similarity Score', fontsize=20)
    fig.supxlabel('Rotation Angle', fontsize=20)
    # Adjust layout and add a supertitle
    fig.suptitle(f'Similarity of 0° compared to other Rotations with n_b:{key[1]}', fontsize=16, y=1.01)
    # plt.tight_layout()
    plt.savefig(file_name, bbox_inches='tight')
    
def loops_plot_for_mean_similarity(state_paths, configs_paths, n_b, parent_path):
    sparsity_values = sorted(set(key[0] for key in state_paths.keys()))
    det_values = sorted(set(key[2] for key in state_paths.keys()))
    determ_similarities = defaultdict(dict)
    corr_similarities = defaultdict(dict)
    for d in det_values:
        print('Calculating similarities for deterministic:', d)
        for s in sparsity_values:
            print('Calculating similarities for sparsity:', s)
            repeat_values = sorted(set(key[3] for key in state_paths.keys() if key[0] == s and key[2] == d and n_b == key[1]))
            for r in repeat_values:
                determ_similarities[d] |= mean_similarity_pipeline((s, n_b, d, r), state_paths, configs_paths).items()
                # set_trace()
                corr_similarities[d] |= corr_similarity_pipeline((s, n_b, d, r), state_paths, configs_paths).items()
    # set_trace()
    plot_all_activations_true_false(determ_similarities['True'], determ_similarities['False'], metric='mean', file_name=f'{parent_path}mean_similarities_plot_n_b_{n_b}.png')
    plot_all_activations_true_false(corr_similarities['True'], corr_similarities['False'], metric='mean', file_name=f'{parent_path}corr_similarities_plot_n_b_{n_b}.png')
    print(f'Plots saved at {parent_path}mean_similarities_plot_n_b_{n_b}.png')

def file_check(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    return file_path


def main():
    results_path = make_plots_folder("/home/users/MTrappett/mtrl/BranchGatingProject/branchNetwork/data/rl_gumbel_sl_comparison_plots/similarity_plots/")
    sl_path = '/home/users/MTrappett/mtrl/BranchGatingProject/branchNetwork/data/sl_determ_gates/'
    rl_path = '/home/users/MTrappett/mtrl/BranchGatingProject/branchNetwork/data/rl_gumbel/'
    loop_ray = ray.remote(loops_plot_for_mean_similarity)
    loops = []
    n_branches = [1] #, 2, 7, 14, 28, 49, 98]
    # ray.init(num_cpus=10)
    for path in [rl_path, sl_path]: # , rl_path
        # set_trace()
        if path == rl_path:
            parser_name = rl_filename_parser
            folder_name = 'RL_plots'
        else:
            parser_name = parse_filename 
            folder_name = 'SL_plots'
        state_paths, config_paths = load_state_models(path, parser_name)
        # set_trace()
        for n_b in n_branches:
            # loops.append(loop_ray.remote(state_paths, config_paths, n_b, file_check(f'{results_path}/{folder_name}/')))
            loops_plot_for_mean_similarity(state_paths, config_paths, n_b, file_check(f'{results_path}/{folder_name}_'))
    # ray.get(loops)
    # ray.shutdown()


if __name__=='__main__':
    main()