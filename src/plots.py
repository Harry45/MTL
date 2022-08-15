"""
Description: Generate plots for Galaxy Zoo.

Author: Arrykrishna Mootoovaloo
Date: August 2022
Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
Project: One/Few-Shot Learning for Galaxy Zoo
"""
import os
import pandas as pd
import matplotlib.pylab as plt

# our script and functions
import settings as st

plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'sans-serif', 'serif': ['Palatino']})
FIGSIZE = (12, 8)
FONTSIZE = 25


def calc_percent_task(table: pd.DataFrame, task_number: int = 1) -> pd.Series:
    """Calculate the percentage of objects in each task.

    Args:
        table (pd.DataFrame): The table to calculate the percentage.
        task_number (int, optional): The task number. Defaults to 1.

    Returns:
        pd.Series: the percentage of objects in each task.
    """

    # sum across columns
    column_sum = table[st.LABELS['task_' + str(task_number)]].sum(0)

    # calculate percentage
    percentage = column_sum / column_sum.sum() * 100

    return percentage


def plot_pie(percentage: list, index: int = 0, save: bool = False):
    """Plot the pie chart for the percentage of objects in each task.

    Args:
        percentage (list): a list containing the percentage for each task.
        index (int, optional): The ith task [0, 9]. Defaults to 0.
        save (bool, optional): Option to save the plot. Defaults to False.
    """

    # number of tasks
    nobject = len(percentage[index].values)

    # different choice of colors
    colors = iter([plt.cm.Pastel1(k) for k in range(nobject)])  # pylint: disable=maybe-no-member

    # plot the pie chart
    _, ax1 = plt.subplots(figsize=(12, 12))
    _, texts, autotexts = ax1.pie(
        percentage[index].values, labels=percentage[index].keys(),
        autopct='%1.1f%%', shadow=False, colors=colors)
    plt.setp(texts, fontsize=FONTSIZE)
    plt.setp(autotexts, fontsize=FONTSIZE)

    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    ax1.axis('equal')

    if save:
        path = 'plots/pie/'
        os.makedirs(path, exist_ok=True)
        plt.savefig(path + 'pie_' + str(index) + '.pdf', bbox_inches='tight')
        plt.savefig(path + 'pie_' + str(index) + '.png', bbox_inches='tight')
        plt.close()

    else:
        plt.show()
