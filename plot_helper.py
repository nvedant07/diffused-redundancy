import matplotlib as mpl
# mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib_venn import venn3, venn2
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

import seaborn as sns
import numpy as np
import os, sys, warnings

from pathlib import Path

RESULTS_FOLDER_NAME = 'results'

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf', '#7532a8']


def set_paper_friendly_params():
    plt.style.use('seaborn-paper')
    plt.rcParams['font.size'] = 24
    plt.rcParams['axes.labelsize'] = 32
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.linewidth'] = 0.2
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 28
    plt.rcParams['ytick.labelsize'] = 28
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['figure.titlesize'] = 25
    plt.rcParams['lines.linewidth'] = 4.0
    plt.rcParams['lines.markersize'] = 12
    plt.rcParams['lines.markeredgewidth'] = 3
    plt.rcParams['grid.color'] = 'grey'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 100


def line_plot(lines_y, x_title, y_title, plot_title, subfolder, filename, extension='png', x_vals=None, 
    legend_vals=None, vertical_line=None, 
    horizontal_lines=None, horizontal_lines_err=None, colors=None, linestyles=None,
    y_lims=None, root_dir='.', paper_friendly_plots=False, plot_inside=False, legend_location='best', savefig=True, figsize=(5,3), 
    marker=False, results_subfolder_name='untitled', grid_spacing=None, y_err=None, legend_ncol=None, 
    inset=None):
    """
    Custom function to make a line plot.
    lines_y: list of lists or a 2D numpy array. Each list/row contains y_coordinates for a particular line.
    x_title: x-axis label
    y_title: y-axis label
    plot_title: Plot title
    filename, subfolder, extension: to be saved as <root_dir>/results/<subfolder>/<filename>.<extension>
    x_vals: x-coordinates for all the y-coordinates of different lines, if None then will be assumed to be 
            [1,2,3,...,len(lines_y[0])], if values are strings then x_vals are taken as the tick labels
    legend_vals: a string corresponding to each line
    colors: a list containing color strings for each of the line in lines_y and also for vertical line (if vertical line is specified)
    linestyles: a list containing linestyles for each of the line in lines_y. No need to specify linestyle for vertical line.
    y_err: a list containing errors in each line in lines_y
    """

    if paper_friendly_plots:
        extension = 'pdf'
        set_paper_friendly_params()
    else:
        sns.set_style('whitegrid')

    if savefig:
        Path('{}/{}/{}/{}'.format(root_dir, RESULTS_FOLDER_NAME, results_subfolder_name, subfolder)).mkdir(parents=True, exist_ok=True)
    
    if x_vals is None:
        x_vals = [np.arange(1, len(lines_y[i]) + 1) for i in range(len(lines_y))]
    if not (isinstance(x_vals[0], list) or isinstance(x_vals[0], tuple) or isinstance(x_vals[0], np.ndarray)):
        x_vals = [x_vals] * len(lines_y)
    
    assert np.all([len(x_vals[i]) == len(lines_y[i]) for i in range(len(lines_y))]), \
        "All lists in (x_vals, lines_y) should be of the same size"

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    if grid_spacing is not None:
        ax.grid(b=True, color='#acacac', which='major', linestyle=':', linewidth=grid_spacing)
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    if not paper_friendly_plots:
        ax.set_title(plot_title)

    if y_lims is not None:
        ax.set_ylim(y_lims)

    max_xv, max_xv_len = None, None
    for i in range(len(x_vals)):
        if isinstance(x_vals[i][0], np.str_) or isinstance(x_vals[i][0], str):
            if max_xv_len is None or max_xv_len < len(x_vals[i]):
                max_xv_len = len(x_vals[i])
                max_xv = x_vals[i]
            x_vals[i] = np.arange(1, len(x_vals[i]) + 1)
    if max_xv is not None:
        ax.set_xticks(np.arange(1, max_xv_len + 1))
        ax.set_xticklabels(max_xv, rotation='vertical')
        print ('Set x_vals')

    for i in range(len(lines_y)):
        ax.plot(x_vals[i], lines_y[i], 
            color=COLORS[i] if colors is None else colors[i], 
            marker='o' if isinstance(marker[i], bool) and marker[i] else \
                marker[i] if marker is not None else None, 
            alpha=0.75,
            linestyle=linestyles[i] if linestyles is not None else '-', 
            label=legend_vals[i] if legend_vals is not None else "")
    
    if y_err is not None:
        assert len(y_err) == len(lines_y)
        for i in range(len(y_err)):
            ax.fill_between(x_vals[i], np.array(lines_y[i]) - np.array(y_err[i]), 
                np.array(lines_y[i]) + np.array(y_err[i]), alpha=0.15, color=COLORS[i] if colors is None else colors[i])

    if vertical_line is not None:
        for j in range(len(vertical_line)):
            ax.axvline(x=vertical_line[j], c=COLORS[len(lines_y)+j] if colors is None else colors[len(lines_y)+j], 
                linestyle='--', label=legend_vals[len(lines_y)+j] if len(legend_vals) > len(lines_y)+j else "")
        ax.tick_params('x', which='minor', direction='in', pad=-12)
        ax.xaxis.set_ticks(vertical_line, minor=True)
        if isinstance(vertical_line[0], float):
            vertical_line = ['{:.2f}'.format(x) for x in vertical_line]
        # ax.xaxis.set_ticklabels(list(map(str, vertical_line)), minor=True)
    else:
        vertical_line = [] # dirty workaround
    
    if horizontal_lines is not None:
        for j in range(len(horizontal_lines)):
            ax.axhline(y=horizontal_lines[j], c=COLORS[len(lines_y)+len(vertical_line)+j] if colors is None else colors[len(lines_y)+len(vertical_line)+j], 
                linestyle='--', label=legend_vals[len(lines_y)+len(vertical_line)+j] if len(legend_vals) > len(lines_y)+len(vertical_line)+j else "")
        if horizontal_lines_err is not None:
            for j in range(len(horizontal_lines_err)):
                ax.fill_between(x_vals[i], horizontal_lines[j] - horizontal_lines_err[j], 
                    horizontal_lines[j] + horizontal_lines_err[j], alpha=0.15, 
                    color=COLORS[len(lines_y)+len(vertical_line)+j] if colors is None else colors[len(lines_y)+len(vertical_line)+j])


    figpath = "{}/{}/{}/{}/{}.{}".format(root_dir, RESULTS_FOLDER_NAME, results_subfolder_name, subfolder, filename, extension)
    wiki_figpath = "{}/{}/{}/{}.{}".format(RESULTS_FOLDER_NAME, results_subfolder_name, subfolder, filename, extension)
    
    if legend_vals is not None:
        if not paper_friendly_plots:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(bbox_to_anchor=(1, 0.5))
        elif paper_friendly_plots and plot_inside:
            ax.legend(loc=legend_location, ncol=1 if legend_ncol is None else legend_ncol)

    if inset is not None:
        axins = zoomed_inset_axes(ax, zoom=inset['zoom'], loc=inset['loc'])
        filtered_xvals, filtered_yvals = [], []
        all_ins_y = []
        for x, y in zip(x_vals, lines_y):
            filt_xy = [(x_, y_) for x_, y_ in zip(x,y) \
                if x_ <= inset['xlim'][1] and x_ >= inset['xlim'][0]]
            filt_x, filt_y = list(zip(*filt_xy))
            filtered_xvals.append(filt_x)
            filtered_yvals.append(filt_y)
            all_ins_y.extend(filt_y)
        axins.set_ylim((np.min(all_ins_y) - 0.05, np.max(all_ins_y) + 0.05))
        axins.set_xlim(inset['xlim'])
        axins.set_yticklabels([])
        axins.set_xticklabels([])
        axins.set_yticks([])
        axins.set_xticks([])
        for i in range(len(filtered_yvals)):
            axins.plot(filtered_xvals[i], filtered_yvals[i], 
                color=COLORS[i] if colors is None else colors[i], 
                marker='o' if isinstance(marker[i], bool) and marker[i] else \
                    marker[i] if marker is not None else None, 
                alpha=0.75,
                linestyle=linestyles[i] if linestyles is not None else '-')
        mark_inset(ax, axins, loc1=inset['loc1'], loc2=inset['loc2'], fc="none", ec="#4C4E52")

    if savefig:
        plt.savefig(figpath, bbox_inches='tight', dpi=fig.dpi, transparent=True)
        plt.show()
    else:
        plt.show()
    
    if legend_vals is not None and paper_friendly_plots and not plot_inside:
        fig_legend = plt.figure(figsize=(3, 3))
        handles, labels = ax.get_legend_handles_labels()
        fig_legend.legend(handles, labels, 'center', ncol=1 if legend_ncol is None else legend_ncol)
        fig_legend.savefig("{}/{}/{}/{}/{}_legend.{}".format(root_dir, 
            RESULTS_FOLDER_NAME, results_subfolder_name, subfolder, 
            filename, extension), bbox_inches='tight')

    
    plt.clf()
    plt.close()
    return wiki_figpath


def line_plot_two_y_axis(ax1_lines, ax2_lines, x_title, y_titles, plot_title, subfolder, filename, extension='png', x_vals=None, 
    legend_vals=None, vertical_line=None, vertical_lines_labels=None, horizontal_line=None, colors=None, linestyles=None, 
    root_dir='.', paper_friendly_plots=False, plot_inside=False, legend_location='best', savefig=True, figsize=(5, 3), marker=False,
    results_subfolder_name='untitled', ax1_lines_err=None, ax2_lines_err=None, y_lims1=None, y_lims2=None):
    """
    Custom function to make a line plot.
    lines_y: list of lists or a 2D numpy array. Each list/row contains y_coordinates for a particular line.
    x_title: x-axis label
    y_titles: list of size two, y-axis label for ax1 and ax2 respectively
    plot_title: Plot title
    filename, subfolder, extension: to be saved as <root_dir>/results/<subfolder>/<filename>.<extension>
    x_vals: x-coordinates for all the y-coordinates of different lines, if None then will be assumed to be 
            [1,2,3,...,len(lines_y[0])], if values are strings then x_vals are taken as the tick labels
    legend_vals: a string corresponding to each line
    colors: a list containing color strings for each of the line in lines_y and also for vertical line (if vertical line is specified)
    linestyles: a list containing linestyles for each of the line in lines_y. No need to specify linestyle for vertical line.
    ax1_lines_err: error bars on ax1 lines
    ax2_lines_err: error bars on ax2 lines
    """

    if paper_friendly_plots:
        extension = 'pdf'
        set_paper_friendly_params()
    else:
        sns.set_style('whitegrid')

    if savefig:
        Path('{}/{}/{}/{}'.format(root_dir, RESULTS_FOLDER_NAME, results_subfolder_name, subfolder)).mkdir(parents=True, exist_ok=True)


    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.grid(b=True, color='#acacac', which='major', linestyle=':', linewidth=0.3)
    ax1.set_xlabel(x_title)
    ax1.set_ylabel(y_titles[0])
    if y_lims1:
        ax1.set_ylim(y_lims1)
    if not paper_friendly_plots:
        ax1.set_title(plot_title)

    if x_vals is None:
        x_vals = np.arange(1, len(ax1_lines[0]) + 1)
        x_ticklabels = [str(x) for x in x_vals]
    else:
        if isinstance(x_vals[0], np.str_) or isinstance(x_vals[0], str):
            x_ticklabels = x_vals
            x_vals = np.arange(1, len(x_vals) + 1)
        else:
            x_ticklabels = None

    l1 = []
    for i in range(len(ax1_lines)):
        l1 += ax1.plot(x_vals, ax1_lines[i], color=COLORS[i] if colors is None else colors[i], 
            marker='o' if marker else '', alpha=0.75,
            linestyle = linestyles[i] if linestyles is not None else '-', 
            label=legend_vals[i] if legend_vals is not None else "")
    
    if x_ticklabels is not None:
        ax1.set_xticks(x_vals)
        ax1.set_xticklabels(x_ticklabels)

    if vertical_line is not None:
        for j in range(len(vertical_line)):
            ax1.axvline(x=vertical_line[j], c=COLORS[len(ax1_lines)+j] if colors is None else colors[len(ax1_lines)+j], 
                linestyle='--', label=vertical_lines_labels[j] if vertical_lines_labels is not None else "")
        # ax1.tick_params('o', which='minor', direction='in', pad=-12)
        ax1.xaxis.set_ticks(vertical_line, minor=True)
        if isinstance(vertical_line[0], float):
            vertical_line = ['{:.2f}'.format(x) for x in vertical_line]
        ax1.xaxis.set_ticklabels(list(map(str, vertical_line)), minor=True)
        # x_tics = ax.xaxis.get_majorticklocs()
        # x_tics = np.append(x_tics, vertical_line)
        # x_tics = np.sort(x_tics)
        # ax.xaxis.set_ticks(x_tics)
    else:
        vertical_line = [] # dirty workaround; I am sorry :(
    
    if horizontal_line is not None:
        for j in range(len(horizontal_line)):
            ax1.axhline(y=horizontal_line[j], 
                c=COLORS[len(ax1_lines)+len(vertical_line)+j] if colors is None else colors[len(ax1_lines)+len(vertical_line)+j], 
                linestyle='--', linewidth=4.0)
    else:
        horizontal_line = [] # dirty workaround; I am sorry :(
    
    ## error bars
    if ax1_lines_err is not None:
        assert len(ax1_lines_err) == len(ax1_lines)
        for i in range(len(ax1_lines_err)):
            ax1.fill_between(x_vals, np.array(ax1_lines[i]) - np.array(ax1_lines_err[i]), 
                np.array(ax1_lines[i]) + np.array(ax1_lines_err[i]), alpha=0.15, color=COLORS[i] if colors is None else colors[i])
    

    ax2 = ax1.twinx()
    ax2.grid(False)
    ax2.set_ylabel(y_titles[1])
    if y_lims2:
        ax2.set_ylim(y_lims2)
    
    l2 = []
    for i in range(len(ax2_lines)):
        l2 += ax2.plot(x_vals, ax2_lines[i], 
            color=COLORS[len(ax1_lines) + len(vertical_line) + len(horizontal_line) + i] \
                if colors is None else colors[len(ax1_lines) + len(vertical_line) + len(horizontal_line) + i], 
            marker='o' if marker else '', alpha=0.75,
            linestyle = linestyles[len(ax1_lines) + i] if linestyles is not None else '-', 
            label=legend_vals[len(ax1_lines) + i] if legend_vals is not None else "")
    
    ## error bars
    if ax2_lines_err is not None:
        assert len(ax2_lines_err) == len(ax2_lines)
        for i in range(len(ax2_lines_err)):
            ax2.fill_between(x_vals, np.array(ax2_lines[i]) - np.array(ax2_lines_err[i]), 
                np.array(ax2_lines[i]) + np.array(ax2_lines_err[i]), alpha=0.15, 
                color=COLORS[len(ax1_lines) + len(vertical_line) + len(horizontal_line) + i] if colors is None else \
                      colors[len(ax1_lines) + len(vertical_line) + len(horizontal_line) + i])

    figpath = "{}/{}/{}/{}/{}.{}".format(root_dir, RESULTS_FOLDER_NAME, results_subfolder_name, subfolder, filename, extension)
    wiki_figpath = "{}/{}/{}/{}.{}".format(RESULTS_FOLDER_NAME, results_subfolder_name, subfolder, filename, extension)
    
    if legend_vals is not None:
        if not paper_friendly_plots or (paper_friendly_plots and plot_inside):
            lns = l1 + l2
            labels = [l.get_label() for l in lns]
            ax1.legend(lns, labels, loc=legend_location)
    
    fig.tight_layout()

    if savefig:
        plt.savefig(figpath, bbox_inches='tight')
        plt.show()
    else:
        plt.show()
    
    if legend_vals is not None and paper_friendly_plots and not plot_inside:
        fig_legend = plt.figure(figsize=(3, 3))
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        fig_legend.legend(handles1 + handles2, labels1 + labels2, 'center', ncol=1)
        fig_legend.savefig("{}/{}/{}/{}/{}_legend.{}".format(root_dir, 
            RESULTS_FOLDER_NAME, results_subfolder_name, subfolder, 
            filename, extension), bbox_inches='tight')

    
    plt.clf()
    plt.close()
    return wiki_figpath



def image_plot(image_object, results_dir, filename, extension='png'):
    sns.set_style('white')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    ax.imshow(image_object if image_object.shape[-1] == 3 else image_object.reshape((image_object.shape[0], image_object.shape[1])),
        cmap='viridis' if image_object.shape[-1] == 3 else 'grey',
        interpolation="bilinear")
    fig.savefig('{}/{}.{}'.format(results_dir, filename, extension))
    plt.close()
    return '{}/{}.{}'.format(results_dir, filename, extension)

def stitched_images(image_objects, plot_titles, results_dir, filename, extension, 
    global_title='', columns=5, savefig=True, plot_title_colors=None, figsize=(15,10), paper_friendly_plots=False):
    
    if paper_friendly_plots:
        extension = 'pdf'
        set_paper_friendly_params()
        sns.set_style('white')
    else:
        sns.set_style('white')

    if savefig:
        Path(results_dir).mkdir(exist_ok=True, parents=True)

    if plot_title_colors is not None:
        assert len(plot_title_colors) == len(plot_titles)

    columns = columns if len(image_objects) > columns else len(image_objects)
    rows = int(len(image_objects)/columns)
    fig = plt.figure(figsize=figsize)
    for i in range(1, columns * rows + 1):
        ax = fig.add_subplot(rows, columns, i)
        # ax.axis('off')
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.imshow(image_objects[i-1] if image_objects[i-1].shape[-1] == 3 else \
            image_objects[i-1].reshape((image_objects[i-1].shape[0], image_objects[i-1].shape[1])), 
            cmap='viridis' if image_objects[i-1].shape[-1] == 3 else 'gray', interpolation='bilinear')
        ax.set_title(plot_titles[i-1], color=plot_title_colors[i-1] if plot_title_colors is not None else 'black')
    fig.suptitle(global_title)
    if savefig:
        plt.savefig('{}/{}.{}'.format(results_dir, filename, extension))
        print ("Saved fig at {}/{}.{}".format(results_dir, filename, extension))
        plt.show()
    else:
        plt.show()
    plt.close()
    return '{}/{}.{}'.format(results_dir, filename, extension)


def stitched_bar(bars, plot_titles, results_dir, filename, extension, y_lims, x_vals, legend_vals=None, 
    global_title='', columns=5, savefig=True, plot_title_colors=None, figsize=(15,10)):
    sns.set_style('white')
    if savefig:
        Path(results_dir).mkdir(parents=True, exist_ok=True)

    if plot_title_colors is not None:
        assert len(plot_title_colors) == len(plot_titles)

    x = np.arange(1, len(x_vals[0]) + 1)
    w = 0.2

    columns = columns if len(bars) > columns else len(bars)
    rows = int(len(bars)/columns)
    fig = plt.figure(figsize=figsize)
    if isinstance(y_lims[0], float) or isinstance(y_lims[0], int):
        y_lims = [y_lims] * len(bars)
    for i in range(1, columns * rows + 1):
        ax = fig.add_subplot(rows, columns, i)
        ax.set_ylim(y_lims[i-1])
        for j in range(len(bars[i-1])):
            # print (x + j*w, bars[i-1][j])
            ax.bar(x + j*w, bars[i-1][j], width=w, color=COLORS[j], align='center', label=legend_vals[j] if legend_vals is not None else '')
        ax.set_xticks(x + (len(bars[i - 1]) - 1) * w/2)
        ax.set_xticklabels(x_vals[i - 1], rotation='vertical')
        ax.legend(loc='best')
        ax.set_title(plot_titles[i-1], fontsize=10, color=plot_title_colors[i-1] if plot_title_colors is not None else 'black')
    fig.suptitle(global_title)
    if savefig:
        fig.savefig('{}/{}.{}'.format(results_dir, filename, extension))
    plt.show()
    plt.close()
    return '{}/{}.{}'.format(results_dir, filename, extension)

def plot_histograms(histogram_vals, x_title, y_title, plot_title, subfolder, filename, extension='png', x_vals=None, 
    legend_vals=None, hist_type='usual', show_fig=False, results_subfolder_name='untitled', figsize=(10,10)):
    """
    histogram_vals: list of arrays where each array represents a sequence which needs to be plotted as histogram
    legend_vals: name of legend items. len(legend_vals) == len(histogram_vals)
    type: 'usual' or 'stacked'. 'usual' would make each histogram bar sum upto one. 'stacked' is essentially a bar plot where each bar sums upto 1
    """
    sns.set_style('white')
    assert len(legend_vals) == len(histogram_vals)
    num_bin = 10
    min_error, max_error = None, None
    for error_vals in histogram_vals:
        if len(error_vals) == 0:
            continue
        if min_error is None or min(error_vals) < min_error:
            min_error = min(error_vals)
        if max_error is None or max(error_vals) > max_error:
            max_error = max(error_vals)

    if min_error is None or max_error is None: # This can only happen if all histogram_vals are empty (happens for train set)
        return 

    bin_lims = np.linspace(min_error,max_error,num_bin+1)
    bin_centers = 0.5*(bin_lims[:-1]+bin_lims[1:])
    bin_widths = bin_lims[1:]-bin_lims[:-1]

    # For the stacked option
    bin_totals = None
    for i in range(len(histogram_vals)):
        hist, x_ticks = np.histogram(histogram_vals[i], bins=bin_lims)
        bin_totals = hist if bin_totals is None else bin_totals + hist

    new_histogram_vals = histogram_vals.copy()
    for i in range(len(histogram_vals)):
        hist, x_ticks = np.histogram(histogram_vals[i], bins=bin_lims)
        if hist_type == 'usual': 
            new_histogram_vals[i] = hist/len(histogram_vals[i])
        elif hist_type == 'stacked':
            assert len(bin_totals) == len(hist)
            new_histogram_vals[i] = hist/bin_totals
        else:
            raise ValueError('{} not  a valid hist_type'.format(hist_type))
    Path('{}/{}/{}'.format(RESULTS_FOLDER_NAME, results_subfolder_name, subfolder)).mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.grid(b=True, color='#ACACAC', which='major', linestyle=':', linewidth=0.3)
    if x_vals is not None:
        x_tick_pos = np.arange(1, len(new_histogram_vals[0]) + 1)
        ax.set_xticks(x_tick_pos)
        ax.set_xticklabels(list(map(str, x_vals)))

    # ax.hist(histogram_vals, bins=50, label=legend_vals, density=True, histtype='step', alpha=0.5)
    cum_size = np.zeros(len(new_histogram_vals[0]))
    for i in range(len(new_histogram_vals)):
        ax.bar(bin_centers, new_histogram_vals[i], bottom=cum_size, width = bin_widths, align = 'center', alpha = 0.5, label=legend_vals[i])
        cum_size += np.array(new_histogram_vals[i])

    ax.legend(loc='best')
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.set_title(plot_title)
    figpath = "{}/{}/{}/{}.{}".format(RESULTS_FOLDER_NAME, results_subfolder_name, subfolder, filename, extension)
    plt.savefig(figpath, bbox_inches='tight')

    if show_fig:
        plt.show()

    plt.clf()
    plt.close()
    return figpath

def plot_heatmaps(maps, x_labels, y_labels, plot_title="", subplot_titles=None, subfolder="heatmaps", filename="default", file_format='png', vmin=0, vmax=1, show_fig=False, cols=None,
    x_title='', y_title='', annotate=False, types=None, paper_friendly_plots=False, annotation_fontsize=12, root_dir='.', figsize=(8,6), results_subfolder_name='untitled'):
    """
    maps: a list of 2D arrays. Eac 2D array represents a heatmap/image.
    types: default is None. Each 2D array in maps is considered a heatmap if types is None. Else needs to be a list of length len(maps) where each entry is either 'maps' or 'image' 
    x_labels: labels to give to x_ticks
    y_labels: labels to give to y_ticks
    plot_title: Global plot title
    subplot_titles: Titles to give to each heatmap, in same order as maps
    """
    if paper_friendly_plots:
        file_format = 'pdf'
        set_paper_friendly_params()
    else:
        sns.set_style('white')

    Path('{}/{}/{}'.format(RESULTS_FOLDER_NAME, results_subfolder_name, subfolder)).mkdir(parents=True, exist_ok=True)
    if types is None:
        types = ['maps'] * len(maps)
    # if not 'maps' in types:
    #     raise ValueError("Atleast one entry needs to ebe a heatmap! Else use stitched_images!")
    
    if cols is None:
        columns, rows = 1, len(maps)
    else:
        columns, rows = cols, int(len(maps)/cols)
        
    if columns * rows > 1 and (vmin != 0 or vmax != 1):
        warnings.warn("vmin and vmax should be set to 0 and 1 respectively when number of subplots are > 1!")
    fig = plt.figure(figsize=figsize)
    for i in range(1, columns * rows + 1):
        ax = fig.add_subplot(rows, columns, i)
        if x_labels is None and y_labels is None:
            ax.axis('off')

        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)

        ax.set_title(subplot_titles[i-1] if subplot_titles is not None else "")
        if x_labels is not None:
            ax.set_xticks(np.arange(len(x_labels)))
            ax.set_xticklabels(x_labels)
        if y_labels is not None:
            ax.set_yticks(np.arange(len(y_labels)))
            if paper_friendly_plots and results_subfolder_name == 'cifar10_automated_all_labels':
                ax.set_yticklabels(["" for label in y_labels])
            else:
                ax.set_yticklabels(y_labels)

        im = ax.imshow(maps[i-1], vmin=vmin, vmax=vmax, cmap=plt.cm.BuPu) if types[i-1] == 'maps' else \
            ax.imshow(maps[i-1] if maps[i-1].shape[-1] == 3 else maps[i-1].reshape((maps[i-1].shape[0], maps[i-1].shape[1])))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90) # rotate x tick labels by 90 degrees

        if annotate:
            fmt_str = '{:.2f}' if maps[i-1].dtype == float else '{:.0f}'
            thresh = (np.nanmin(maps[i-1]) + np.nanmax(maps[i-1])) / 2.
            for k in range(maps[i-1].shape[0]):
                for j in range(maps[i-1].shape[1]):
                    ax.text(j, k, fmt_str.format(maps[i-1][k, j]),
                            ha="center", va="center", fontsize=annotation_fontsize,
                            color="white" if maps[i-1][k, j] > thresh else "black")
    
    if not paper_friendly_plots:
        fig.suptitle(plot_title)

    if 'maps' not in types:
        pass
    else:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

    figpath = '{}/{}/{}/{}/{}.{}'.format(root_dir, RESULTS_FOLDER_NAME, results_subfolder_name, subfolder, filename, file_format)
    plt.savefig(figpath, bbox_inches='tight')
    if show_fig:
        plt.show()
    plt.close()

    return figpath

def plot_venn(sets, set_labels, subfolder, filename, extension='png', title='', results_subfolder_name='untitled'):
    Path('{}/{}/{}'.format(RESULTS_FOLDER_NAME, results_subfolder_name, subfolder)).mkdir(parents=True, exist_ok=True)
    if len(sets) != 2 and len(sets) != 3:
        raise ValueError("Only supported for 2 or 3 sets")

    if len(sets) == 2:
        venn2(sets, set_labels)
    else:
        vd = venn3(subsets=(1, 1, 1, 1, 1, 1, 1), set_labels=set_labels) # To get all circles of same size
        a_b_c = sets[0].intersection(sets[1].intersection(sets[2]))
        vd.get_label_by_id('100').set_text('{}'.format(len(sets[0] - sets[1] - sets[2]))) # Only Set A
        vd.get_label_by_id('010').set_text('{}'.format(len(sets[1] - sets[0] - sets[2]))) # Only Set B
        vd.get_label_by_id('001').set_text('{}'.format(len(sets[2] - sets[0] - sets[1]))) # Only Set C
        vd.get_label_by_id('110').set_text('{}'.format(len(sets[0].intersection(sets[1]) - a_b_c))) # Only Set A^B
        vd.get_label_by_id('011').set_text('{}'.format(len(sets[1].intersection(sets[2]) - a_b_c))) # Only Set B^C
        vd.get_label_by_id('101').set_text('{}'.format(len(sets[0].intersection(sets[2]) - a_b_c))) # Only Set A^C
        vd.get_label_by_id('111').set_text('{}'.format(len(a_b_c))) # Set A^B^C

    
    figpath = '{}/{}/{}/{}.{}'.format(RESULTS_FOLDER_NAME, results_subfolder_name, subfolder, filename, extension)
    plt.title(title)
    plt.savefig(figpath, bbox_inches='tight')
    plt.clf()
    plt.close()

    return figpath

def bar_plot(bars, x_title, y_title, plot_title, subfolder, x_labels, filename, legend_vals=None, 
    savefig=True, show_fig=True, extension='png', x_tick_colors=None, horizontal_lines=None, figsize=(8,6), colors=None, hatchstyles=None,
    y_lims=None, paper_friendly_plots=False, results_folder_name=RESULTS_FOLDER_NAME, results_subfolder_name='untitled', legend_ncol=None):
    
    if paper_friendly_plots:
        extension = 'pdf'
        set_paper_friendly_params()
    
    Path('{}/{}/{}'.format(results_folder_name, results_subfolder_name, subfolder)).mkdir(parents=True, exist_ok=True)

    x = np.arange(1, len(x_labels) + 1)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    w = 0.2
    
    if y_lims is not None:
        # If ylims are specified, they take precendence
        ax.set_ylim(y_lims)
    elif np.array(bars).shape[0] > 0 and np.array(bars).shape[1] > 0:
        minimum_val, max_val = np.min(np.array(bars)), np.max(np.array(bars))
        ax.set_ylim(minimum_val - 0.01, max_val + 0.01)
    
#     print (bars, legend_vals)
    legend_patches = []
    for i in range(len(bars)):
        ax.bar(x + i*w, bars[i], width=w, color=COLORS[i] if colors is None else colors[i], 
            hatch=hatchstyles[i] if hatchstyles is not None else '',
            align='center', label=legend_vals[i] if legend_vals is not None else '', alpha=0.99)
        legend_patches.append(
            mpatches.Patch(facecolor=COLORS[i] if colors is None else colors[i],
                           hatch=hatchstyles[i] if hatchstyles is not None else '',
                           label=legend_vals[i] if legend_vals is not None else '',
                           alpha=1.))
    
    if horizontal_lines is not None:
        for j in range(len(horizontal_lines)):
            ax.axhline(y=horizontal_lines[j], c=COLORS[len(bars)+j], linestyle='--')
    
    if legend_vals is not None and not paper_friendly_plots:
        ax.legend(handles=legend_patches, loc='best', bbox_to_anchor=(1, 0.5))
    
    ax.set_title(plot_title)
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.set_xticks(x + (len(bars) - 1) * w/2)
    ax.set_xticklabels(x_labels, rotation='vertical')
    
    if x_tick_colors is not None:
        [i.set_color(x_tick_colors[idx]) for idx, i in enumerate(plt.gca().get_xticklabels())]
    
    figpath = '{}/{}/{}/{}.{}'.format(results_folder_name, results_subfolder_name, subfolder, filename, extension)
    if savefig:
        plt.savefig(figpath, bbox_inches='tight')
        plt.show()
    if show_fig:
        plt.show()

    if paper_friendly_plots:
        fig_legend = plt.figure(figsize=(3, 3))
        fig_legend.legend(handles=legend_patches, loc='center', ncol=1 if legend_ncol is None else legend_ncol)
        fig_legend.savefig('{}/{}/{}/{}_legend.{}'.format(results_folder_name, 
            results_subfolder_name, subfolder, filename, extension), bbox_inches='tight')
    
    plt.clf()
    plt.close()
    return figpath

def scatter_plot(x, y, labels, plot_title, subfolder, filename, x_label, y_label, extension='png', figsize=(5,3), x_ticklabels=None,
    x_err=None, y_err=None, legend_vals=None, paper_friendly_plots=False, colors=None, markers=None, sizes=None, 
    results_subfolder_name='untitled', legend_ncol=None):
    
    Path('{}/{}/{}'.format(RESULTS_FOLDER_NAME, results_subfolder_name, subfolder)).mkdir(exist_ok=True, parents=True)

    if paper_friendly_plots:
        extension = 'pdf'
        set_paper_friendly_params()

    fig = plt.figure(figsize=figsize)
    # plt.xscale('symlog')
    ax = fig.add_subplot(111)
    
    if not isinstance(x[0], list) and not isinstance(x[0], np.ndarray):
        assert isinstance(x[0], int) or isinstance(x[0], float) or isinstance(x[0], np.int64) or isinstance(x[0], np.float64)
        ax.scatter(x, y, c='tab:orange', label=legend_vals, alpha=0.75, edgecolors='none')
    else:
        assert legend_vals is not None and len(legend_vals) == len(x) and len(legend_vals) == len(y)
        for idx, (x_val, y_val) in enumerate(zip(x, y)):
            if x_err is not None or y_err is not None:
                ax.errorbar(x_val, y_val, xerr=x_err[idx] if x_err is not None else None, yerr=y_err[idx] if y_err is not None else None,
                    c=COLORS[idx] if colors is None else colors[idx], marker=markers[idx] if markers is not None else 'o', 
                    label=legend_vals[idx], alpha=0.65, markersize=5 if sizes is None else sizes[idx], markeredgecolor='none', 
                    fmt=markers[idx] if markers is not None else 'o')
            else:
                ax.scatter(x_val, y_val, c=COLORS[idx] if colors is None else colors[idx], 
                    marker=markers[idx] if markers is not None else 'o', 
                    label=legend_vals[idx], alpha=0.65, s=50 if sizes is None else sizes[idx], edgecolors='none')

    if labels is not None:
        for i, txt in enumerate(labels):
            ax.annotate(txt, (x[i], y[i]))

    ax.grid(True)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if x_ticklabels is not None:
        ax.set_xticks(np.arange(len(x_ticklabels)))
        ax.set_xticklabels(x_ticklabels, rotation='vertical')

    if not paper_friendly_plots:
        # ax.legend(loc='best')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(bbox_to_anchor=(1, 0.5))

    plt.title(plot_title)

    figpath = '{}/{}/{}/{}.{}'.format(RESULTS_FOLDER_NAME, results_subfolder_name, subfolder, filename, extension)    
    plt.savefig(figpath, bbox_inches='tight')

    if paper_friendly_plots:
        fig_legend = plt.figure(figsize=(3, 3))
        handles, labels = ax.get_legend_handles_labels()
        fig_legend.legend(handles, labels, 'center', ncol=1 if legend_ncol is None else legend_ncol)
        fig_legend.savefig('{}/{}/{}/{}_legend.{}'.format(RESULTS_FOLDER_NAME, results_subfolder_name, 
            subfolder, filename, extension), bbox_inches='tight')

    return figpath

