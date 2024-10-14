# Description: This file contains the functions to generate the plots for the analysis pipeline.
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipdb import set_trace


def plot_comparison_scatter_matrix(sl_data_dict: dict, rl_data_dict: dict, n_bs: list, sparsity_values: list, error_metric: str='sem'):
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

    # Define colors for easier reference
    color_axis1 = 'blue'
    color_axis2 = 'darkred'

    # Create subplot titles for the top of each column
    column_titles = [n_branches[j] for j in range(len(n_branches))]  # One title per column
    # Create subplots with secondary y-axes specified
    fig = make_subplots(rows=len(gating_type), cols=len(n_branches), shared_xaxes=True,
                        vertical_spacing=0.07, horizontal_spacing=0.02, column_titles=column_titles,
                        specs=[[{'secondary_y': True} for _ in n_branches] for _ in gating_type],
                        x_title='Sparsity')

    for i, gate_bool in enumerate(gating_type):
        gate_name = 'Deterministic<br>' if gate_bool == 'True' else 'Stochastic<br>'
        for j, n_b in enumerate(n_branches):
            row = i + 1
            col = j + 1
            # Generate data and error bounds
            sl_data_rem = sl_data_dict[gate_bool][n_b]['rem']
            sl_error_rem = sl_data_dict[gate_bool][n_b][f'rem_{error_metric}']
            sl_data_ft = sl_data_dict[gate_bool][n_b]['ft']
            sl_error_ft = sl_data_dict[gate_bool][n_b][f'ft_{error_metric}']
            rl_data_rem = rl_data_dict[gate_bool][n_b]['rem']
            rl_error_rem = rl_data_dict[gate_bool][n_b][f'rem_{error_metric}']
            rl_data_ft = rl_data_dict[gate_bool][n_b]['ft']
            rl_error_ft = rl_data_dict[gate_bool][n_b][f'ft_{error_metric}']
            # set_trace()
            # Add traces for primary and secondary y-axis
            fig.add_traces([
                # SL, Remembering
                go.Scatter(
                    x=sparsity_values, y=sl_data_rem, mode='lines+markers',
                    line=dict(color=color_axis1, width=2, dash='solid'),
                    name='', legendgroup='method1', showlegend=False,
                ),
                # go.Scatter(x=sparsity_values+sparsity_values[::-1],y=sl_error_rem+sl_error_rem[::-1],
                #            fill='toself', fillcolor=color_axis1, hoverinfo='skip', showlegend=False,),
                # RL, Remembering
                go.Scatter(
                    x=sparsity_values, y=rl_data_rem, mode='lines+markers',
                    line=dict(color=color_axis1, width=2, dash='dash'),
                    name='', legendgroup='method2', showlegend=False,
                ),
                # go.Scatter(x=sparsity_values+sparsity_values[::-1],y=rl_error_rem+rl_error_rem[::-1],
                #             fill='toself', fillcolor=color_axis1, hoverinfo='skip', showlegend=False,),
                # SL, Forward Transfer
                go.Scatter(
                    x=sparsity_values, y=sl_data_ft, mode='lines+markers',
                    line=dict(color=color_axis2, width=2, dash='solid'),
                    name='', legendgroup='method1', showlegend=False,
                ),
                # go.Scatter(x=sparsity_values+sparsity_values[::-1],y=sl_error_ft+sl_error_ft[::-1],
                #             fill='toself', fillcolor=color_axis2, hoverinfo='skip', showlegend=False,),
                # RL, Forward Transfer
                go.Scatter(
                    x=sparsity_values, y=rl_data_ft, mode='lines+markers',
                    line=dict(color=color_axis2, width=2, dash='dash'),
                    name='', legendgroup='method2', showlegend=False,
                ),
                # go.Scatter(x=sparsity_values+sparsity_values[::-1],y=rl_error_ft+rl_error_ft[::-1],
                #             fill='toself', fillcolor=color_axis2, hoverinfo='skip', showlegend=False,),
            ], rows=row, cols=col, secondary_ys=[False, False, True, True])

            # Customize y-axes
            fig.update_yaxes(title_text=f'{gate_name}' if j == 0 else '', showgrid=False,
                                row=row, col=col, secondary_y=False, color=color_axis1, title_font_color='black',
                                showticklabels=(j == 0 or j==len(n_branches)-1), ticks='inside', title_standoff=25)
            
            fig.update_yaxes(title_text=f'', showgrid=False,
                                row=row, col=col, secondary_y=True, color=color_axis2, title_font_color='black',
                                showticklabels=(j == len(n_branches) - 1 or j == 0), ticks='inside', title_standoff=25)

            fig.update_xaxes(showgrid=False, row=row, col=col, ticks='inside')

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

    # fig.show()
    return fig


