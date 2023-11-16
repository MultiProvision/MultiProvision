import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from functools import reduce
from matplotlib.patches import Patch


def plot_time(export_name, batch_name):
    dir_stats = f"results/{export_name}/stats/{batch_name}"

    df = pd.read_csv(dir_stats, index_col=0)

    names_list = df.columns.tolist()
    time_list = []
    for name in names_list:
        if name[-4:] == 'time':
            time_list.append(name)

    times = []
    for name in time_list:
        times.append(df[[name]])
    df_times = pd.concat(times, axis='columns')

    df_times = df_times.round(2)
    max_y = max(df_times.max().tolist())*1.3
    ax = df_times.plot(kind='bar', title=f"{export_name} para {batch_name[:-4]}")
    plt.ylabel("Time (s)")
    plt.xlabel("Schedulers")
    for p in ax.patches:
        if p.get_height() != 0:
            ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()), rotation=90.0)
    ax.set_ylim([0, max_y])

    dir_save = f"plots/{export_name}/time/"
    os.makedirs(dir_save, exist_ok=True)
    plt.savefig(dir_save + batch_name[:-4] + ".png", dpi=400, bbox_inches='tight')
    plt.close("all")


def plot_energy(export_name, batch_name):
    dir_stats = f"results/{export_name}/stats/{batch_name}"

    df = pd.read_csv(dir_stats, index_col=0)

    names_list = df.columns.tolist()
    energy_list = []
    for name in names_list:
        if name[-6:] == 'energy':
            energy_list.append(name)

    energy = []
    for name in energy_list:
        energy.append(df[[name]])
    df_energy = pd.concat(energy, axis='columns')

    df_energy = df_energy.round(2)
    max_y = max(df_energy.max().tolist()) * 1.3
    ax = df_energy.plot(kind='bar', title=f"{export_name} para {batch_name[:-4]}")
    plt.ylabel("Energy (J)")
    plt.xlabel("Schedulers")
    for p in ax.patches:
        if p.get_height() != 0:
            ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()), rotation=90.0)
    ax.set_ylim([0, max_y])

    dir_save = f"plots/{export_name}/energy/"
    os.makedirs(dir_save, exist_ok=True)
    plt.savefig(dir_save + batch_name[:-4] + ".png", dpi=400, bbox_inches='tight')
    plt.close("all")


def plot_timeline(export_name, batch_name, scheduler_name, lines):
    dir_timeline = f"results/{export_name}/timeline/{batch_name[:-4]}/{scheduler_name}.csv"
    df_timeline = pd.read_csv(dir_timeline, index_col=0)

    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(df_timeline.columns.tolist())))

    ax = df_timeline.plot(kind='barh', stacked=True, title=scheduler_name,
                          width=1, color=colors, figsize=(15, len(lines)-1))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), fancybox=True, shadow=True, ncol=10)

    plt.xlabel("Time")
    plt.xticks(lines)
    ax.set_xlim(0, max(lines))
    plt.ylim(-0.5, (len(lines)-1) - 0.5)

    dir_save = f"plots/{export_name}/timeline/{batch_name[:-4]}/"
    os.makedirs(dir_save, exist_ok=True)
    plt.savefig(dir_save + scheduler_name + ".png", dpi=400, bbox_inches='tight')
    plt.close("all")


def all_results(export_name, batches):

    dir_open = f"results/{export_name}/stats/"
    dataframe_name = pd.read_csv(dir_open + batches[0], index_col=0).columns.tolist()
    device_name_list = []
    for name in dataframe_name:
        if name[-4:] == 'time':
            device_name_list.append(name[:-5])

    lista_time = []
    for batch in batches:
        df = pd.read_csv(dir_open + batch, index_col=0)

        names_list = df.columns.tolist()
        time_list = []

        for name in names_list:
            if name[-4:] == 'time':
                time_list.append(name)
        times = []
        for name in time_list:
            df1 = df[[name]].round(2)
            df1.columns = [batch[:-4]]
            df1.index = ["Only Cpu", "Only Gpu", "FCFS", "rr", "wrr", "Max Min", "Min Min", "Fist Fit"]
            times.append(df1)
        lista_time.append(times)
    df_time = []

    for i in range(len(times)):
        select = []
        for element in lista_time:
            select.append(element[i])
        df_time.append(pd.concat(select, axis='columns').transpose())

    list_energy = []
    for batch in batches:
        df = pd.read_csv(dir_open + batch, index_col=0)

        names_list = df.columns.tolist()
        time_list = []

        for name in names_list:
            if name[-6:] == 'energy':
                time_list.append(name)
        times = []
        for name in time_list:
            df1 = df[[name]].round(2)
            df1.columns = [batch[:-4]]
            df1.index = ["Only Cpu", "Only Gpu", "FCFS", "rr", "wrr", "Max Min", "Min Min", "Fist Fit"]
            times.append(df1)
        list_energy.append(times)
    df_energy = []

    for i in range(len(times)):
        select = []
        for element in list_energy:
            select.append(element[i])
        df_energy.append(pd.concat(select, axis='columns').transpose())

    df_energy_sum = reduce(lambda a, b: a.add(b, fill_value=0), df_energy)

    list_energy_plot = []
    aux = 0
    for df in df_energy:
        aux = aux + df
        list_energy_plot.append(aux)

    list_style = [" ", "////"]

    fig, axes = plt.subplots(nrows=2, ncols=1)
    fig.suptitle(f'Hardware: {export_name}', fontsize=16)

    for i in range(len(device_name_list)):
        fig.legend(
            handles=[Patch(facecolor="white", edgecolor="black", hatch=list_style[i], label=device_name_list[i], )],
            bbox_to_anchor=(0, i * 0.028))
    max_y_time = 0.0

    for value in df_time:
        if max(value.max().tolist()) >= max_y_time:
            max_y_time = max(value.max().tolist()) * 1.3

    max_y_energy = max(df_energy_sum.max().tolist()) * 1.3

    i = 0
    for df in df_time:
        ax = df.plot(kind='bar', title="Makespan", ax=axes[0], ylabel="Time (s)", figsize=(10, 10), edgecolor="black",
                     hatch=list_style[i], width=0.8)
        i += 1
        ax.legend(df_time[0].columns.tolist())
        ax.set_ylim([0, max_y_time])

    j = len(list_energy_plot) - 1

    for _ in list_energy_plot:
        ax = list_energy_plot[j].plot(kind='bar', title="Energy", ax=axes[1], ylabel="Energy (KJ)", figsize=(10, 10),
                                      edgecolor="black", hatch=list_style[j], width=0.8)

        ax.get_legend().remove()
        ax.set_ylim([0, max_y_energy])
        current_values = ax.get_yticks()

        for i, label in enumerate(ax.get_yticklabels()):
            label.set_text(str(current_values[i] / 1000))

        j -= 1
        i += 1

    plt.tight_layout()

    dir_save = f"plots/{export_name}/all_results/"
    os.makedirs(dir_save, exist_ok=True)
    plt.savefig(dir_save + export_name + ".png", dpi=400, bbox_inches='tight')
    plt.close("all")
