from math import ceil
from typing import List, Tuple

import PIL
import plotly
import plotly.express as px
from plotly.subplots import make_subplots


def plot_wordclouds(list_tuples_title_wordcloud:List[Tuple[str, PIL.Image.Image]]):
    '''
    list_tuples_title_wordcloud: in the form of: [('author0', pil_img_wordcloud0), ('author1', pil_img_wordcloud1), ...]
    '''
    from math import ceil

    import matplotlib.pyplot as plt

    num_authors = len(list_tuples_title_wordcloud)
    num_cols_plot_wordcloud = 2
    num_rows_plot_wordcloud = ceil(num_authors / num_cols_plot_wordcloud)

    fig, axes = plt.subplots(num_rows_plot_wordcloud, num_cols_plot_wordcloud, figsize=(18, 10))

    idx = 0
    for i in range(num_rows_plot_wordcloud):
        for j in range(num_cols_plot_wordcloud):
            if idx == num_authors:
                break
            ax = axes[i, j]
            ax.imshow(list_tuples_title_wordcloud[idx][1])
            ax.axis('off');
            ax.set_title(list_tuples_title_wordcloud[idx][0], fontsize=30)
            idx += 1




def plot_vertical_histograms_sidebyside(
    list_tuples_titles_histogram_figures: List[Tuple[str, plotly.graph_objs.Figure]],
    height=1000,
    width=1200
):
    num_row_plots = 1
    num_col_plots = len(list_tuples_titles_histogram_figures)
    fig = make_subplots(
        rows=num_row_plots,
        cols=num_col_plots,
        subplot_titles=([t[0] for t in list_tuples_titles_histogram_figures])
    )
    
    for i in range(num_col_plots):
        subfig_data = list_tuples_titles_histogram_figures[i][1].data
        assert len(subfig_data) == 1
        fig.add_trace(
            subfig_data[0],
            row=1, col=i+1
        )
        
        fig.update_yaxes(row=1, col=i+1, autorange='reversed')
        
    fig.update_layout(
        height=height,
        width=width,
        yaxis_title='Word',
        xaxis_title='Count',
    )

    return fig


def plot_most_common_words_freq(list_tuples_from_counter, height=1000, width=500, orientation='h'):
    if orientation == 'h':
        y = [x[0] for x in list_tuples_from_counter]
        x = [x[1] for x in list_tuples_from_counter]
    elif orientation == 'v':
        height, width = width, height
        y = [x[1] for x in list_tuples_from_counter]
        x = [x[0] for x in list_tuples_from_counter]
    else:
        raise ValueError('Orientation has to be `h` or `v`')
    fig = px.bar(
        y=y,
        x=x,
        orientation=orientation
    ).update_layout(
        xaxis=dict(tickfont=dict(size=8)),
        height=height,
        width=width,
        yaxis_title='Count',
        xaxis_title='Word'
    )
    
    if orientation == 'h':
        fig.update_layout(
        yaxis_title='Word',
        xaxis_title='Count',
        yaxis=dict(
            autorange='reversed'
        )
    )
    
    return fig
