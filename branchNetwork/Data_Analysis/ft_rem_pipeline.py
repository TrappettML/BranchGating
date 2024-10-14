import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.subplots as sp
import plotly.express as px
from collections import defaultdict
import matplotlib.pyplot as plt
import scipy.stats as st
import os
import pickle
import re
import warnings
from ipdb import set_trace
from branchNetwork.Data_Analysis.Analysis_pipeline import plot_comparison_scatter_matrix

from branchNetwork.Data_Analysis.DetermGates_correlation import parse_filename


# new metrics for FT and Remembering

def ft_rem_metric(auc_values, train_order):
    # remembing is the mean of the area under the curve for all tasks 
    # after the trainng task divided by the AUC for the training task block.
    # FT is the mean of the area under the curve for all tasks before the training task
    rem_list = []
    ft_list = []
    for (_, eval), auc in auc_values.items():
        eval_block_index = train_order.index(eval)
        eval_train_auc = auc_values[(eval, eval)]
        for train in train_order:
            if train_order.index(train) < eval_block_index:
                ft_list.append(auc)
            elif train_order.index(train) > eval_block_index:
                rem_list.append(auc/eval_train_auc)
    return np.mean(ft_list), np.mean(rem_list)


def prep_accuracy_metric(data_dict_acc, key):
    """"
    Take the accuracy data for a model key and return the auc for each train,eval task pair
    inputs: data_dict_acc: dictionary of accuracy data
            key: model key
    outputs: auc_values: dictionary of auc for each train, eval task pair
             train_order: order of training tasks
    """
    acc_data = data_dict_acc[key]
    train_order = [i['train_task'] for i in acc_data['first_last_data']]
    eval_acc_dict = data_dict_acc[key]['task_evaluation_acc']
    trapz_dict = {}
    for train in train_order:
        train_key = f'training_{train}'
        for k,v in eval_acc_dict[train_key].items():
            # print(v)
            trapz_dict[(train, k)] = np.trapz(v)
    return trapz_dict, train_order


def parse_filename_accuracies(filename, matching_pattern=False):
    if not matching_pattern:
        pattern1 = r"si_all_task_accuracies_learn_gates_False_soma_func_sum_l2_0.0_lr_0.0001_n_npb_([^_]*)_([^_]*)_n_branches_([^_]*)_([^_]*)_sparsity_([^_]*)_hidden_784_784_det_masks_([^_]*)_model_name_BranchModel_repeat_([^_]*)_epochs_per_task_20.pkl"
    # Try matching with the first pattern
    matches = re.search(pattern1, filename)
    # Extract values and convert to appropriate types
    sparsity = float(matches.group(5))
    n_branch =int(matches.group(3))
    n_npb = int(matches.group(1))
    repeat = int(matches.group(7))
    det_mask = str(matches.group(6))
    # lr = float(matches.group(5))
    repeat = int(matches.group(7))
    # set_trace()
    return (sparsity, n_branch, det_mask, repeat)
        
def rl_filename_parser(filename):
    pattern = r"si_all_task_accuracies_soma_func_sum_lr_0.0001_loss_func_RLCrit_n_npb_([^_]*)_([^_]*)_n_branches_([^_]*)_([^_]*)_sparsity_([^_]*)_det_masks_([^_]*)_model_name_BranchModel_repeat_([^_]*)_epochs_per_task_20.pkl"
    matches = re.search(pattern, filename)
    sparsity = float(matches.group(5))
    n_branch = int(matches.group(3))
    n_npb = int(matches.group(1))
    repeat = int(matches.group(7))
    det_mask = str(matches.group(6))
    return (sparsity, n_branch, det_mask, repeat)


def load_all_accuracies(directory, parser=parse_filename_accuracies):
    # all_acc = load_all_accuracies(path)
    # Dictionaries to hold data from all pickle files
    accuracies = {}
    # Loop through each file in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # print(file)
            # print(filename)
            if file.endswith('.pkl'):
                # Construct the full path to the file
                file_path = os.path.join(root, file)
                if 'accuracies' in file_path:
                    # Parse the filename into a tuple of variables and determine the data type
                    # print(f'{file_path=}')
                    key = parser(file_path)
                    # print(f'filename: {file_path}; \n{key=}')
                    if key is not None:
                        # Open and load the pickle file
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                            # Store the data 
                            accuracies[key] = data
    return accuracies


def path_to_auc(directory: str, parser=parse_filename_accuracies) -> tuple:
    """Takes in the path with subdirectories containing pickle files
       outputs the auc values for each model in a dictionary and training order
       inputs: directory: path to the directory containing the pickle files
       outputs: model_auc_values[(sparsity, etc.)]=[auc's]: dictionary of auc values for each model
                train_order: order of training tasks
       """
    all_acc = load_all_accuracies(directory, parser)
    model_auc_values = {}
    for key in all_acc.keys():
        model_auc_values[key], train_order = prep_accuracy_metric(all_acc, key)
    return model_auc_values, train_order


def aggregate_experiment_runs(data: dict, train_order: list) -> dict:
    """
    Process the data from a single experiment into a dictionary 
    where the key is a tuple of the experiment parameters and the 
    values are the statistics of the metrics FT and remembering.
    input: data: dictionary where key=(parameters) and value=()
    """
    aggregated_data = defaultdict(list)

    # Aggregate results by (a, b, c, d, e) ignoring the repetition index
    for (sparsity, n_branch, det_mask, repeat), auc_values in data.items():
        aggregated_data[(sparsity, n_branch, det_mask, repeat, 'forward_transfer')], aggregated_data[(sparsity, n_branch, det_mask, repeat, 'remembering')] = ft_rem_metric(auc_values, train_order)

    return aggregated_data


def make_2d_matrix(data_dict: dict) -> tuple:
    # Extract unique sorted lists of sparsity and number of branches
    n_branch_values = sorted(set(key[1] for key in data_dict.keys()))
    sparsity_values = sorted(set(key[0] for key in data_dict.keys()))
    det_mask = sorted(set(key[2] for key in data_dict.keys()))
    repeat_values = sorted(set(key[3] for key in data_dict.keys()))
    # Populate the matrices
    heat_map_data = {}
    for b in det_mask:
        # Initialize matrices for the heatmaps
        transfer_matrix = np.full((len(sparsity_values), len(n_branch_values), len(repeat_values)), np.nan)
        remembering_matrix = np.full((len(sparsity_values), len(n_branch_values), len(repeat_values)), np.nan)
        # Populate the matrices
        for (sparsity, n_branch, det_mask, repeat, _metric), stats in data_dict.items():
            # if repeat == 5:
            i = sparsity_values.index(sparsity)
            j = n_branch_values.index(n_branch)
            k = repeat_values.index(repeat)
            # print(f'{i=}, {j=}, {k=}, {metric=}, {stats=}')
            if _metric == 'forward_transfer':
                transfer_matrix[i, j, k] = stats
            elif _metric == 'remembering':
                remembering_matrix[i, j, k] = stats

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            remembering_matrix_mean = np.nanmean(remembering_matrix, axis=2, out=None)
            transfer_matrix_mean = np.nanmean(transfer_matrix, axis=2, out=None)
            rem_med_matrix = np.nanmedian(remembering_matrix, axis=2, out=None)
            ft_med_matrix = np.nanmedian(transfer_matrix, axis=2, out=None)
            rem_sem_matrix = st.sem(remembering_matrix, axis=2)
            ft_sem_matrix = st.sem(transfer_matrix, axis=2)
            rem_std_matrix = np.nanstd(remembering_matrix, axis=2, out=None)
            ft_std_matrix = np.nanstd(transfer_matrix, axis=2, out=None)

        heat_map_data[b] = {'ft': transfer_matrix_mean, 'rem': remembering_matrix_mean, 
                            'rem_med': rem_med_matrix, 'ft_med': ft_med_matrix,
                            'rem_sem': rem_sem_matrix, 'ft_sem': ft_sem_matrix,
                            'rem_std': rem_std_matrix, 'ft_std': ft_std_matrix}
    # set_trace()
    return heat_map_data, sparsity_values, n_branch_values


def plot_heatmap(matrix_d: dict, title: str, y_labels: list, x_labels: list, color_bar_title: str, metric: str):
    for b, matrix in matrix_d.items():
        fig = go.Figure(data=go.Heatmap(
            z=matrix[metric],
            x=x_labels,
            y=y_labels,
            colorscale='Viridis',
            colorbar=dict(title=f'{color_bar_title}', titlefont=dict(size=26), tickfont=dict(size=26)),
            xgap=1,  # Adjust gaps between cells if necessary
            ygap=1
        ))
        fig.update_layout(
            title=f'{title}<br>Deterministic Gating: {b}',
            yaxis=dict(title='Sparsity', type='category'),
            xaxis=dict(title='# Branches', type='category'),
            autosize=False,
            width=1100,
            height=600,
            xaxis_tickfont=dict(size=26),
            yaxis_tickfont=dict(size=26),
            xaxis_title_font=dict(size=32),  # Sets the X-axis title font size
            yaxis_title_font=dict(size=32)
        )
        return fig
        
def parse_data(path, parser=parse_filename_accuracies) -> dict:
    auc_dict, train_order = path_to_auc(path, parser)
    aggregated_data = aggregate_experiment_runs(auc_dict, train_order) 
    return aggregated_data

def pipeline_heatmap(aggregated_data: dict) -> list:
    heat_map_data, sparsity_values, n_branch_values = make_2d_matrix(aggregated_data)
    figs = [plot_heatmap(heat_map_data, f'Forward Transfer', sparsity_values, n_branch_values,  'FT AUC', 'ft'),
            plot_heatmap(heat_map_data, f'Remembering', sparsity_values, n_branch_values,  'Rem Ratio', 'rem'),
            plot_heatmap(heat_map_data, f'Forward Transfer - Median', sparsity_values, n_branch_values,  'FT AUC', 'ft_med'),
            plot_heatmap(heat_map_data, f'Remembering - Median', sparsity_values, n_branch_values,  'Rem Ratio', 'rem_med')]
    return figs

def expand_heatmap_dicts(heatmap_dict: dict, n_branch_values) -> dict:
    """heatmap dict: [determ_gating_bool][metric or error] = matrix (sparsity x n_branches)
    """
    expanded_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for b in heatmap_dict.keys():
        for metric in heatmap_dict[b].keys():
            for n_b in n_branch_values:
                expanded_dict[b][n_b][metric] = heatmap_dict[b][metric][:,n_branch_values.index(n_b)]
    
    return expanded_dict

    
def make_comparison_plots(sl_heatmap_dict: dict, rl_heatmap_dict: dict, sparsity_values: list, n_branch_values: list)-> go.Figure:
    """heatmap: dict where keys=determ_gating_bool, values={'ft': matrix, 'rem': matrix, 'rem_med': matrix, 'ft_med': matrix})
        sparsity is 0 index, n_branch is 1 index.     
    """
    sl_expanded = expand_heatmap_dicts(sl_heatmap_dict, n_branch_values)
    rl_expanded = expand_heatmap_dicts(rl_heatmap_dict, n_branch_values)
    # set_trace()
    comp_fig = plot_comparison_scatter_matrix(sl_expanded, rl_expanded, n_branch_values, sparsity_values, error_metric='std')
    return comp_fig

def write_fig(fig: go.Figure, filename: str):
    with open(filename, 'w') as f:
        f.write(fig.to_html(full_html=True, include_plotlyjs='cdn'))
    print(f'Fig saved to {filename}')

def comp_fig_pipeline(sl_path: str, rl_path: str, save_path: str):
    sl_auc_dict, sl_train_order = path_to_auc(sl_path)
    rl_auc_dict, rl_train_order = path_to_auc(rl_path, parser=rl_filename_parser)
    sl_aggregated_data = aggregate_experiment_runs(sl_auc_dict, sl_train_order)
    rl_aggregated_data = aggregate_experiment_runs(rl_auc_dict, rl_train_order)
    sl_heat_map_data, sparsity_values, n_branch_values = make_2d_matrix(sl_aggregated_data)
    rl_heat_map_data, _, _ = make_2d_matrix(rl_aggregated_data)
    comp_fig = make_comparison_plots(sl_heat_map_data, rl_heat_map_data, sparsity_values, n_branch_values)
    write_fig(comp_fig, f'{save_path}/comparison_rl_sl_fig.html')


def make_plots_folder(folder_name: str):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

def prep_data_for_plotting(data_dict_acc, key, repeats: list):
    all_data = []
    for r in repeats:
        key = key[:-1] + (r,)
        acc_data = data_dict_acc[key]
        train_order = [i['train_task'] for i in acc_data['first_last_data']]
        joined_accuracies = defaultdict(list)
        eval_acc_dict = data_dict_acc[key]['task_evaluation_acc']
        for train in train_order:
            train_key = f'training_{train}'
            for k,v in eval_acc_dict[train_key].items():
                joined_accuracies[k] += v
        all_data.append(joined_accuracies)
    mean_data = {k: None for k in joined_accuracies.keys()}
    error_data = {k: None for k in joined_accuracies.keys()}
    for k in mean_data.keys():
        stacked = np.stack([d[k] for d in all_data])
        mean_data[k] = np.mean(stacked, axis=0)
        # error_data[k] = st.sem(stacked, axis=0)
        error_data[k] = np.std(stacked, axis=0)
    return joined_accuracies, error_data, train_order

def plot_train_sequence_accuracies(dict_of_accuracies, error_dict, task_order, g=20, model_info=None, save_fig=False):
    # Create the figure
    fig = go.Figure()

    # Add traces for the lists
    length = 0
    _min, _max = float('inf'), float('-inf')
    colors = {k: px.colors.qualitative.Plotly[i] for i,k in enumerate(dict_of_accuracies.keys())}
    # set_trace()
    for k,v in dict_of_accuracies.items():
        # print(f'{v=}')
        fig.add_trace(go.Scatter(y=v, mode='lines+markers', line=dict(width=5), marker=dict(size=11, color=str(colors[k])), name=f'Eval Task {k}'))
        fig.add_trace(go.Scatter(y=v+error_dict[k], mode='lines', marker=dict(color=str(colors[k])), line=dict(width=0), showlegend=False, opacity=0.5, hoverinfo='skip', fill='tonexty'))
        fig.add_trace(go.Scatter(y=v-error_dict[k], mode='lines', marker=dict(color=str(colors[k])), line=dict(width=0), showlegend=False, opacity=0.5, fill='tonexty',))
        length = len(v) if len(v) > length else length
        _min = min(_min, min(v))
        _max = max(_max, max(v))

    fig.add_annotation(x=-5, y=1.095, text=f"Angle:",
                            showarrow=False, xref='x', yref='paper', font=dict(color="grey", size=30))
    # Add vertical lines and annotations
    for t,i in enumerate(range(0, length, g)):
        if i > 0:  # Avoid drawing a line at the very start
            # Add a vertical line
            fig.add_shape(type="line", x0=i-0.5, y0=0, x1=i-0.5, y1=1,
                        line=dict(color="grey", width=1, dash="dash"),
                        xref='x', yref='paper')
        
        # Add text annotation
        if i + g <= length:
            # Position text at the center of the segment
            text_position = (i + (i + g)) / 2
            fig.add_annotation(x=text_position, y=1.08, text=f"{task_order[t]}\u00B0",
                            showarrow=False, xref='x', yref='paper', font=dict(color="grey", size=25))

    # Update layout to show the result better
    fig.update_layout(title='Training Accuracy over Epochs' if not model_info else \
                      f'Training Accuracy over Epochs --- Branches: {model_info[1]}, Sparsity: {model_info[0]}',
                    xaxis_title='Epoch',
                    yaxis_title='Accuracy',
                    title_y=.95,
                    width=1700,
                    height=550,
                    xaxis=dict(showgrid=True),
                    yaxis=dict(range=[-2, 102]),
                    legend=dict(
                    font=dict(size=30)  # Sets the legend text font size
                    ),
                    xaxis_tickfont=dict(size=30),
                    yaxis_tickfont=dict(size=30),
                    xaxis_title_font=dict(size=37),  # Sets the X-axis title font size
                    yaxis_title_font=dict(size=37)
                    )
    return fig

def prep_accuracy_heatmap_metric(data_dict_acc, key):
    acc_data = data_dict_acc[key]
    train_order = [i['train_task'] for i in acc_data['first_last_data']]
    eval_acc_dict = data_dict_acc[key]['task_evaluation_acc']
    trapz_dict = {}
    for train_task in train_order:
        train_key = f'training_{train_task}'
        for k,v in eval_acc_dict[train_key].items():
            # print(v)
            trapz_dict[(train_task, k)] = np.trapz(v)
    return trapz_dict, train_order


def plot_accuracy_heatmap(data, train_order, _key, r_=True):
    # Extract unique values for the x and y axes from the keys
    x_values = train_order
    y_values = sorted(set(key[1] for key in data.keys()), reverse=True)

    # Create a 2D matrix to hold the data for the heatmap
    matrix = np.full((len(y_values), len(x_values)), np.nan)  # Initialize with NaNs
    
    # Populate the matrix with data
    for (x, y), value in data.items():
        x_index = x_values.index(x)
        y_index = y_values.index(y)
        matrix[y_index, x_index] = value

    # Calculate the sum of each row and prepare it as a separate column for its own heatmap
    row_sums = np.nansum(matrix, axis=1).reshape(-1, 1)

    # Set up a subplot grid with two columns
    fig = sp.make_subplots(rows=1, cols=2, subplot_titles=("", ""),
                           horizontal_spacing=0.12,  # Adjust spacing to separate the heatmaps visually
                           shared_yaxes=True,
                           column_widths=[0.9, 0.1])

    # Main heatmap
    fig.add_trace(
        go.Heatmap(
            z=matrix,
            x=x_values,
            y=y_values,
            colorscale='Viridis',
            colorbar=dict(title='Task<br>Area', titlefont=dict(size=20), tickfont=dict(size=26), x=0.79),
            xgap=1,
            ygap=1
        ),
        row=1, col=1
    )
    # Heatmap for totals
    fig.add_trace(
        go.Heatmap(
            z=row_sums,
            x=['Sum over<br>all Tasks'],
            y=y_values,
            colorscale='Viridis',
            colorbar=dict(title='Total<br>Area', titlefont=dict(size=20), tickfont=dict(size=26), x=1),
            xgap=1,
            ygap=1
        ),
        row=1, col=2
    )
    fig.update_layout(
        title=f'Area Under Accuracy Curve --- n_branches: {_key[1]}, sparsity: {_key[0]}',
        # xaxis_title='Eval. Tasks',
        # yaxis_title='Current Train Tasks',
        # xaxis2_title='Total',
        autosize=False,
        width=1500,
        height=400,
        xaxis=dict(title='Training Tasks in Order', type='category', tickfont=dict(size=26), title_font=dict(size=32)),
        xaxis2=dict(title='', type='category', tickfont=dict(size=26), title_font=dict(size=32)),
        yaxis=dict(title='Evaluated Tasks', type='category', tickfont=dict(size=26), title_font=dict(size=32)),
    )
    return fig


def training_plots(data_dict_acc, plot_folder):
    # Write figures to the same HTML file
    sorted_keys = sorted(data_dict_acc.keys(), key=lambda x: (x[2], x[0], x[1], x[3]))
    sparsity_values = sorted(set(key[0] for key in data_dict_acc.keys()))
    det_mask = sorted(set(key[2] for key in data_dict_acc.keys()))
    n_branch_values = sorted(set(key[1] for key in data_dict_acc.keys()))
    repeat_values = sorted(set(key[3] for key in data_dict_acc.keys()))
    for d in det_mask:
        for n in n_branch_values:
            c = 0
            for s in sparsity_values:
                prep_data = prep_data_for_plotting(data_dict_acc, (s, n, d, 5), repeat_values)
                fig = plot_train_sequence_accuracies(prep_data[0], prep_data[1], prep_data[2], 20, (s, n, d, 5), save_fig=True)
                if c == 0:
                    with open(f'{plot_folder}/training_plots_determ_{d}_n_b_{n}.html', 'w') as f:
                        f.write(fig.to_html(full_html=True, include_plotlyjs='cdn'))
                        c += 1
                else:
                    with open(f'{plot_folder}/training_plots_determ_{d}_n_b_{n}.html', 'a') as f:
                        f.write(fig.to_html(full_html=False, include_plotlyjs=False))

                    c += 1
                # Heatmap plot
                trapz1, train_order = prep_accuracy_heatmap_metric(data_dict_acc, (s, n, d, 5))
                fig1 = plot_accuracy_heatmap(trapz1, train_order, (s, n, d, 5))
                with open(f'{plot_folder}/training_plots_determ_{d}_n_b_{n}.html', 'a') as f:
                    f.write(fig1.to_html(full_html=False, include_plotlyjs=False))


def main():
    results_path = make_plots_folder("/home/users/MTrappett/mtrl/BranchGatingProject/branchNetwork/data/rl_sl_comparison_plots/")
    sl_path = '/home/users/MTrappett/mtrl/BranchGatingProject/branchNetwork/data/sl_determ_gates/'
    rl_path = '/home/users/MTrappett/mtrl/BranchGatingProject/branchNetwork/data/RL_mean_rule/'
    count = 0
    for path in [sl_path, rl_path]:
        if path == rl_path:
            parse_filename = rl_filename_parser
        else:
            parse_filename = parse_filename_accuracies
        data = parse_data(path, parse_filename)
        figs = pipeline_heatmap(data)
        for i, fig in enumerate(figs):
            if count == 0:
                with open(f'{results_path}/determ_gating_true_false_heatmaps.html', 'w') as f:
                    f.write(fig.to_html(full_html=True, include_plotlyjs='cdn'))
                    print('saved first fig')
            else:
                with open(f'{results_path}/determ_gating_true_false_heatmaps.html', 'a') as f:
                    f.write(fig.to_html(full_html=False, include_plotlyjs=False))
            count += 1
    print('finished heatmap figs')
    for path in [sl_path, rl_path]:
        data_dict_acc = load_all_accuracies(path)
        training_plots(data_dict_acc, results_path)
    print('finished training plots')

    comp_fig_pipeline(sl_path, rl_path, results_path)
       
       
if __name__ == '__main__':
    main()