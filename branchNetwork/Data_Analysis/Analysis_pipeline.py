# Description: This file contains the functions to generate the plots for the analysis pipeline.
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipdb import set_trace
import numpy as np

OPACITY = 0.3
def comparison_plots(sl_data_dict: dict, rl_data_dict: dict, n_bs: list, sparsity_values: list, error_metric: str='std'):
    """
    Plot metrics for SL and RL data.

    Parameters:
    - sl_data_dict (dict): Dictionary containing SL data.
    - rl_data_dict (dict): Dictionary containing RL data.
    - n_bs (list): List of metrics to plot.
    - sparsity_values (list): List of sparsity values.
    """
    # Define colors for SL and RL
    color_sl = 'rgba(0, 0, 255'
    color_rl = 'rgba(255, 0, 0'

    gating_type = ['True', 'False']
    sparsity_values = sparsity_values[:-1]  # Adjust sparsity_values if needed
    # set_trace()
    figs = []  # List to store figures
    metrics = [metric for metric in sl_data_dict.keys() if 'std' not in metric and 'sem' not in metric]  # Get metrics
    rl_metrics = [metric for metric in rl_data_dict.keys() if 'std' not in metric and 'sem' not in metric]  # Get metrics
    # check if all metrics are the same
    for m in metrics:
        assert m in rl_metrics, f"Metric {m} not found in RL data"
    for m_rl in rl_metrics:
        assert m_rl in metrics, f"Metric {m_rl} not found in SL data"

    column_titles = [n_b for n_b in n_bs]  # One title per column
    
    for metric in metrics:
        # Create subplots for each gating type
        fig = make_subplots(
            rows=len(gating_type),
            cols=len(n_bs),
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=0.05,
            horizontal_spacing=0.01, 
            column_titles=column_titles, 
            # specs=[[{'secondary_y': False} for _ in n_bs] for _ in gating_type],
            x_title='Sparsity',
            # subplot_titles=[f'Deterministic Gating' if gt == 'True' else 'Stochastic Gating' for gt in gating_type]
        )
        legend_bool = True
        for i, gate_bool in enumerate(gating_type):
            gate_name = 'Deterministic' if gate_bool == 'True' else 'Stochastic'
            for j, n_b in enumerate(n_bs):
                row = i + 1
                col = j + 1

                # Get data for SL and RL
                sl_data = sl_data_dict[metric][gate_bool][n_b]
                rl_data = rl_data_dict[metric][gate_bool][n_b]
                if 'med' in metric:
                    sl_y_error = np.zeros_like(sl_data)
                    rl_y_error = np.zeros_like(rl_data)
                else:
                    sl_y_error = np.array(sl_data_dict[f'{metric}_{error_metric}'][gate_bool][n_b])
                    rl_y_error = np.array(rl_data_dict[f'{metric}_{error_metric}'][gate_bool][n_b])

                # Plot SL data
                fig.add_trace(
                    go.Scatter(
                        x=sparsity_values,
                        y=sl_data,
                        mode='lines',
                        line=dict(color=f'{color_sl}, 1)', width=2, dash='solid'),
                        name='SL',
                        legendgroup='SL',
                        showlegend=legend_bool,
                    ),
                    row=row,
                    col=col,
                )
                fig.add_trace(go.Scatter(
                    x=sparsity_values,
                    y=sl_data+sl_y_error,
                    mode='lines',
                    line=dict(color=f'{color_sl}, {OPACITY})', width=0),
                    hoverinfo='skip', 
                    showlegend=False,),
                        row=row, col=col)
                                    
                fig.add_trace(go.Scatter(
                    x=sparsity_values,
                    y=sl_data-sl_y_error,
                    mode='lines',
                    line=dict(color=f'{color_sl}, {OPACITY})', width=0),
                    fillcolor=f'{color_sl}, {OPACITY})', 
                    fill='tonexty',
                    hoverinfo='skip',
                    showlegend=False,
                    opacity=0.3),
                        row=row, col=col)

                # Plot RL data
                fig.add_trace(
                    go.Scatter(
                        x=sparsity_values,
                        y=rl_data,
                        mode='lines',
                        line=dict(color=f'{color_rl}, 1)', width=2, dash='dash'),
                        name='RL',
                        legendgroup='RL',
                        showlegend=legend_bool,
                    ),
                    row=row,
                    col=col,
                )
                fig.add_trace(go.Scatter(
                        x=sparsity_values,
                        y=rl_data+rl_y_error,
                        mode='lines',
                        line=dict(color=f'{color_rl}, {OPACITY})', width=0),
                        hoverinfo='skip', 
                        showlegend=False,
                        opacity=0.3),
                            row=row, col=col)
                    
                fig.add_trace(go.Scatter(
                    x=sparsity_values,
                    y=rl_data-rl_y_error,
                    mode='lines',
                    line=dict(color=f'{color_rl}, {OPACITY})', width=0),
                    fillcolor=f'{color_rl}, {OPACITY})', 
                    fill='tonexty',
                    hoverinfo='skip',
                    showlegend=False,
                    opacity=0.3),
                        row=row, col=col)
                
                legend_bool = False
                # Customize y-axis label
                if j == 0:
                    # set_trace()
                    fig.update_yaxes(title_text=f'{gate_name}',
                                    row=row, col=col)
            # fig.update_yaxes(title=f'{gate_name}')
            # fig.update_xaxes(row=row, col=col, ticks='inside')

            # Update layout
            fig.update_layout(
                height=len(gating_type)*250,
                width=len(n_bs)*300,
                title_text=f'Metric: {metric}',
                legend=dict(
                    orientation='h',
                    x=0.5,
                    y=1.05,
                    xanchor='center',
                    yanchor='bottom',
                    font=dict(size=12)
                )
            )
        # Append figure to the list
        figs.append(fig)

    return figs
    

def comparison_plots_2(sl_data_dict: dict, rl_data_dict: dict, n_bs: list, sparsity_values: list, error_metric: str='std'):
    """
    Plot metrics for SL and RL data with models as rows and gating types as traces.

    Parameters:
    - sl_data_dict (dict): Dictionary containing SL data.
    - rl_data_dict (dict): Dictionary containing RL data.
    - n_bs (list): List of metrics to plot.
    - sparsity_values (list): List of sparsity values.
    - error_metric (str): Error metric to exclude from plotting.
    
    Returns:
    - figs (list): List of Plotly figures for each metric.
    """
    # Define colors for SL and RL
    color_det = 'rgba(6, 64, 43'
    color_sto = 'rgba(255, 140, 0'

    # Define gating types
    gating_type = ['True', 'False']
    gating_names = ['Deterministic', 'Stochastic']
    
    # Adjust sparsity_values if needed
    sparsity_values = sparsity_values[:-1]  
    
    figs = []  # List to store figures
    
    # Extract metrics excluding those containing 'std' or 'sem'
    metrics = [metric for metric in sl_data_dict.keys() if 'std' not in metric and 'sem' not in metric]  # Get metrics
    rl_metrics = [metric for metric in rl_data_dict.keys() if 'std' not in metric and 'sem' not in metric]  # Get metrics
    
    # Ensure both SL and RL have the same metrics
    for m in metrics:
        assert m in rl_metrics, f"Metric {m} not found in RL data"
    for m_rl in rl_metrics:
        assert m_rl in metrics, f"Metric {m_rl} not found in SL data"

    column_titles = [str(n_b) for n_b in n_bs]  # One title per column

    # Define row types
    row_types = ['SL', 'RL']
    
    for metric in metrics:
        # Create subplots with 2 rows: SL and RL
        fig = make_subplots(
            rows=2,
            cols=len(n_bs),
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=0.05,
            horizontal_spacing=0.03, 
            subplot_titles=column_titles,
            # y_title='Metric Value',
            x_title='Sparsity'
        )
        
        # Initialize legend display
        legend_shown = True
        
        for i, model_type in enumerate(row_types):
            for j, n_b in enumerate(n_bs):
                row = i + 1
                col = j + 1

                # Select the appropriate data dictionary
                data_dict = sl_data_dict if model_type == 'SL' else rl_data_dict
                # color = color_det if model_type == 'SL' else color_sto
                # dash_style = 'solid' if model_type == 'SL' else 'dash'

                for gate_bool, gate_name in zip(gating_type, gating_names):

                    # Retrieve the corresponding data
                    y_data = np.array(data_dict[metric][gate_bool][n_b])
                    if 'med' in metric:
                        y_error = np.zeros_like(y_data)
                    else:
                        y_error = np.array(data_dict[f'{metric}_{error_metric}'][gate_bool][n_b])
                    color = color_det if gate_bool=='True' else color_sto
                    dash_style = 'solid' if gate_bool=='True' else 'dash'

                    # Add the trace
                    fig.add_trace(
                        go.Scatter(
                            x=sparsity_values,
                            y=y_data,
                            mode='lines',
                            line=dict(color=f'{color}, 1)', width=2, dash=dash_style),
                            name=gate_name,
                            legendgroup=gate_name,
                            showlegend=legend_shown if j == 0 else False,  # Show legend only once per trace
                        ),
                        row=row,
                        col=col
                    )
                    fig.add_trace(go.Scatter(
                        x=sparsity_values,
                        y=y_data+y_error,
                        mode='lines',
                        line=dict(color=f'{color}, {OPACITY})', width=0),
                        hoverinfo='skip', 
                        showlegend=False,
                        opacity=0.3),
                            row=row, col=col)
                    
                    fig.add_trace(go.Scatter(
                        x=sparsity_values,
                        y=y_data-y_error,
                        mode='lines',
                        line=dict(color=f'{color}, {OPACITY})', width=0),
                        fillcolor=f'{color}, {OPACITY})', 
                        fill='tonexty',
                        hoverinfo='skip',
                        showlegend=False,
                        opacity=0.3),
                            row=row, col=col)
                
                # Update y-axis label for the first column
                if j == 0:
                    fig.update_yaxes(title_text=model_type, row=row, col=col)
            
            # After the first column, stop showing legends
            legend_shown = False
        
        # Update the overall layout
        fig.update_layout(
            height=500,  # Adjust as needed
            width=300 * len(n_bs),  # Adjust width based on number of columns
            title_text=f'Metric: {metric}',
            legend=dict(
                orientation='h',
                x=0.5,
                y=1.02,
                xanchor='center',
                yanchor='bottom',
                font=dict(size=12)
            ),
            # template='plotly_white'
        )
        
        # Append the figure to the list
        figs.append(fig)
        # set_trace()
    return figs


def plot_comparison_scatter_matrix(sl_data_dict: dict, rl_data_dict: dict, n_bs: list, sparsity_values: list, error_metric: str='std'):
    """ Dummy Data:
    # Creating dummy data
        sparsity_values = np.linspace(0, 1, 10) # sparsity values, will be categorical.
        gating_type = ['Deterministic<br>Gating', 'Stochastic<br>Gating'] 
        n_branches = ['Returns', 'QVariance', 'ParametersNorm', 'QNorm', 'Srank', 'DormantNeurons'] # replace with n_bs
        data = {game: {metric: np.random.random(sparsity_values.size) for metric in n_branches} for game in gating_type} # SL data
        additional_data = {game: {metric: np.random.random(sparsity_values.size) * 100 for metric in n_branches} for game in gating_type} # RL data
    """
    
    n_branches = n_bs # change to the list of n_b from input
    gating_type = ['True', 'False'] # change to the list of gating_type from input
    sparsity_values = sparsity_values[:-1]
    # Define colors for easier reference
    color_axis1 = 'blue'
    color_axis2 = 'darkred'
    color_axis3 = 'darkorange'
    color_axis4 = 'indianred'

    # Create subplot titles for the top of each column
    column_titles = [n_b for n_b in n_bs]  # One title per column
    # Create subplots with secondary y-axes specified
    fig = make_subplots(rows=len(gating_type), cols=len(n_bs), shared_xaxes=True,
                        vertical_spacing=0.07, horizontal_spacing=0.02, column_titles=column_titles,
                        specs=[[{'secondary_y': True} for _ in n_branches] for _ in gating_type],
                        x_title='Sparsity')
    
    fig2 = make_subplots(rows=len(gating_type), cols=len(n_branches), shared_xaxes=True, shared_yaxes=True,
                        vertical_spacing=0.07, horizontal_spacing=0.02, column_titles=column_titles,
                        specs=[[{'secondary_y': False} for _ in n_branches] for _ in gating_type],
                        x_title='Sparsity')
    
    legend_2 = True
    for i, gate_bool in enumerate(gating_type):
        gate_name = 'Deterministic<br>' if gate_bool == 'True' else 'Stochastic<br>'
        for j, n_b in enumerate(n_branches):
            row = i + 1
            col = j + 1
            # Generate data and error bounds
            # set_trace()
            sl_data_rem = sl_data_dict['remembering'][gate_bool][n_b]
            # sl_error_rem = sl_data_dict[f'remembering_{error_metric}'][gate_bool][n_b]
            sl_data_ft = sl_data_dict['forward_transfer'][gate_bool][n_b]
            sl_error_ft = sl_data_dict[f'forward_transfer_{error_metric}'][gate_bool][n_b]
            rl_data_rem = rl_data_dict['remembering'][gate_bool][n_b]
            rl_error_rem = rl_data_dict[f'remembering_{error_metric}'][gate_bool][n_b]
            rl_data_ft = rl_data_dict['forward_transfer'][gate_bool][n_b]
            rl_error_ft = rl_data_dict[f'forward_transfer_{error_metric}'][gate_bool][n_b]
            # set_trace()
            # Add traces for primary and secondary y-axis
            # fig 1
            fig.add_traces([
                # SL, Remembering
                go.Scatter(
                    x=sparsity_values, y=sl_data_rem, mode='lines',
                    line=dict(color=color_axis1, width=2, dash='solid'),
                    name='', legendgroup='method1', showlegend=False,
                ),
                # go.Scatter(x=np.concatenate([sparsity_values, sparsity_values[::-1]]),
                #            y=np.concatenate([sl_data_rem+sl_error_rem, (sl_data_rem-sl_error_rem)[::-1]]),
                #             fill='toself', fillcolor=color_axis1, hoverinfo='skip', showlegend=False, opacity=0.3),
                # go.Scatter(x=sparsity_values+sparsity_values[::-1],y=sl_error_rem+sl_error_rem[::-1],
                #            fill='toself', fillcolor=color_axis1, hoverinfo='skip', showlegend=False,),
                # RL, Remembering
                go.Scatter(
                    x=sparsity_values, y=rl_data_rem, mode='lines',
                    line=dict(color=color_axis1, width=2, dash='dash'),
                    name='', legendgroup='method2', showlegend=False,
                ),
                # go.Scatter(x=sparsity_values+sparsity_values[::-1],y=rl_error_rem+rl_error_rem[::-1],
                #             fill='toself', fillcolor=color_axis1, hoverinfo='skip', showlegend=False,),
                # SL, Forward Transfer
                go.Scatter(
                    x=sparsity_values, y=sl_data_ft, mode='lines',
                    line=dict(color=color_axis2, width=2, dash='solid'),
                    name='', legendgroup='method1', showlegend=False,
                ),
                # go.Scatter(x=sparsity_values+sparsity_values[::-1],y=sl_error_ft+sl_error_ft[::-1],
                #             fill='toself', fillcolor=color_axis2, hoverinfo='skip', showlegend=False,),
                # RL, Forward Transfer
                go.Scatter(
                    x=sparsity_values, y=rl_data_ft, mode='lines',
                    line=dict(color=color_axis2, width=2, dash='dash'),
                    name='', legendgroup='method2', showlegend=False,
                ),
                # go.Scatter(x=sparsity_values+sparsity_values[::-1],y=rl_error_ft+rl_error_ft[::-1],
                #             fill='toself', fillcolor=color_axis2, hoverinfo='skip', showlegend=False,),
                ], rows=row, cols=col, secondary_ys=[False, False, True, True])
            
            ######## fig 2 ############################################
            sl_data_ttm = sl_data_dict['mean_first_3_acc'][gate_bool][n_b]
            rl_data_ttm = rl_data_dict['mean_first_3_acc'][gate_bool][n_b]
            sl_ttm_error = sl_data_dict[f'mean_first_3_acc_{error_metric}'][gate_bool][n_b]
            rl_ttm_error = rl_data_dict[f'mean_first_3_acc_{error_metric}'][gate_bool][n_b]
            fig2.add_traces([
                go.Scatter(
                    x=sparsity_values, y=sl_data_ttm, mode='lines',
                    line=dict(color=color_axis3, width=2, dash='solid'),
                    name='SL', legendgroup='SL', showlegend=legend_2,
                ),                
                go.Scatter(
                    x=sparsity_values, y=rl_data_ttm, mode='lines',
                    line=dict(color=color_axis4, width=2, dash='dash'),
                    name='RL', legendgroup='RL', showlegend=legend_2,
                )], rows=row, cols=col,)
            legend_2 = False
            # Customize y-axes
            fig.update_yaxes(title_text=f'{gate_name}' if j == 0 else '', showgrid=False,
                                row=row, col=col, secondary_y=False, color=color_axis1, title_font_color='black',
                                showticklabels=(j == 0 or j==len(n_branches)-1), ticks='inside', title_standoff=25)
            
            fig.update_yaxes(title_text=f'', showgrid=False,
                                row=row, col=col, secondary_y=True, color=color_axis2, title_font_color='black',
                                showticklabels=(j == len(n_branches) - 1 or j == 0), ticks='inside', title_standoff=25)

            fig.update_xaxes(showgrid=False, row=row, col=col, ticks='inside')
            ##### custom axis for fig 2
            fig2.update_yaxes(title_text=f'{gate_name}' if j == 0 else '', showgrid=False,
                              row=row, col=col, secondary_y=False)

            fig2.update_xaxes(showgrid=False, row=row, col=col, ticks='inside')


    # Add dummy traces for the legend
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color='black', dash='solid', width=2),
            name='SL Rule',showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color='black', dash='dash', width=2),
            name='RL Rule',showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color=color_axis1, dash='solid', width=2),
            name='Remembering',showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color=color_axis2, dash='solid', width=2),
            name='Forward Transfer',showlegend=True,
        )
    )
    # Master Y-axis labels for primary and secondary axes
    fig.add_annotation(dict(
        text="Remembering Ratio",  # Label for the primary y-axis
        xref="paper", yref="paper",
        x=-0.030, y=0.5,
        showarrow=False,
        textangle=-90,
        font=dict(size=12, color=color_axis1,),
    ))

    fig.add_annotation(dict(
        text="Foward Transfer AUC",  # Label for the secondary y-axis
        xref="paper", yref="paper",
        x=0.97, y=0.5,
        showarrow=False,
        textangle=-90,
        font=dict(color=color_axis2, size=12),   
    ))

    # Adjust layout and display figure
    fig.update_layout(
        height=len(gating_type)*250,
        width=len(n_branches)*300,
        title_text='Comparison of RL and SL Learning Rules',
        legend=dict(
            orientation='h',
            traceorder='normal',
            x=0.45,
            y=1.05,
            xanchor='center',
            yanchor='bottom',
            font=dict(size=14),
        )
    )
        # Adjust layout and display figure
    fig2.update_layout(
        height=len(gating_type)*250,
        width=len(n_branches)*300,
        title_text='Comparison of RL and SL<br>Average over first 3 epochs of final task',
        legend=dict(
            orientation='h',
            traceorder='normal',
            x=0.45,
            y=1.05,
            xanchor='center',
            yanchor='bottom',
            font=dict(size=14),
        )
    )

    # fig.show()
    return fig, fig2


