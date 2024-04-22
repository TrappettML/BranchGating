
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MaxNLocator

from collections import OrderedDict



def make_results_dict(results):
    results_dictionary = dict()
    for result in results:
        results_dictionary[result['model_name']] = result['data']
    # plot_results(results_dictionary)
    return results_dictionary
    
def make_plot(results_dictionary: dict[str, OrderedDict], subfig_labels: list, results_indices: list, title: str, save_str: str, yaxis_label: str='Loss'):
    fig, axs = plt.subplots(len(subfig_labels), 1, figsize=(10, 8), sharex=True, layout="constrained")
    colors ={model_name:color for model_name, color in zip(results_dictionary.keys(), ['blue', 'orange', 'green', 'red'])}
    
    for j, (i, ax) in enumerate(zip(results_indices, axs)):
        for results in results_dictionary.values():
            ax.plot(np.array(list(results.values())[i]), 
                    label=list(results.values())[0], 
                    color=colors[list(results.values())[0]], 
                    linewidth=2)
        
        # Removing the red rectangle by not adding it this time
        ax.grid(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        # ax.set_yticks([0, 1]) 
        
        # Adding vertical bars
        third_of_data = int(len(list(results.values())[i])/3)
        ax.axvline(x=third_of_data, color='grey', linestyle='--', linewidth=2)
        ax.axvline(x=2*third_of_data, color='grey', linestyle='--', linewidth=2)
        
        # Setting larger labels
        ax.set_ylabel(yaxis_label, fontsize=14)

        # Placing larger, bold text aligned with each subplot on the y-axis
        fig.text(1.05, 0.5-(i*0.01), subfig_labels[j], fontsize=12, fontweight='bold',
                verticalalignment='center', horizontalalignment='left', transform=ax.transAxes)

    # Setting the x-axis label only once, with a larger font size
    axs[-1].set_xlabel('Epochs', fontsize=14)

    # Adding a legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    # Adding an overall figure title, a bit larger
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    # plt.savefig(f'/home/users/MTrappett/mtrl/BranchGatingProject/data/plots/{save_str}.png', bbox_inches='tight')
      
      
      
def plot_results(results_dictionary: dict[str, OrderedDict]):
    '''results dictionary will have keys of model name and values of Ordereddict with element:
    ['model_name',
    'Training_Loss', 
    f'd{degrees[0]}_loss', 
    f'd{degrees[1]}_loss',
    f'd{degrees[2]}_loss',
    f'd{degrees[0]}_Accuracy',
    f'd{degrees[1]}_Accuracy',
    f'd{degrees[2]}_Accuracy',]
    '''
    
    subfig_labels = ['Training Loss',
                    'Validation Loss for Task 1', 
                    'Validation Loss for Task 2', 
                    'Validation Loss for Task 3']
    results_indices = [1,2,3,4]
    make_plot(results_dictionary, subfig_labels, results_indices, 'Comparison of all losses during training and evaluation', 'loss_plot', 'Loss')
    
    subfig_labels = ['Training Loss',
                    'Validation Accuracy for Task 1', 
                    'Validation Accuracy for Task 2', 
                    'Validation Accuracy for Task 3']
    results_indices = [1,5,6,7]
    make_plot(results_dictionary, subfig_labels, results_indices,'Comparison of evaluation accuracy after training', 'accuracy_plot', 'Accuracy')
