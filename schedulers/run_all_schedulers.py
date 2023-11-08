import pandas as pd
import argparse as ap
import pathlib as pl
import os
import schedulers as sc
import scheduler_plots as splt


def get_args():
    def check_dir(path):
        p = pl.Path(path)
        if p.is_dir():
            return p
        else:
            raise ap.ArgumentTypeError(f"Directory {path} does not exist")

    def check_file(path):
        p = pl.Path(path)
        if p.is_file():
            return p
        else:
            raise ap.ArgumentTypeError(f"File {path} does not exist")

    def check_int(value):
        try:
            return int(value)
        except ValueError:
            raise ap.ArgumentTypeError(f"{value} is not a valid integer")

    def check_float(value):
        try:
            return float(value)
        except ValueError:
            raise ap.ArgumentTypeError(f"{value} is not a valid floating")

    parser = ap.ArgumentParser(
        description="convert benchmarks result files into a single compiled csv")
    parser.add_argument("-i", dest="info_file",
                        help="input benchmark_data csv file",
                        type=check_file,
                        default=pl.Path("../benchmark_results/benchmark_information.csv"),
                        required=False)
    parser.add_argument("-b", dest="batches_dir",
                        help="directory containing batches csv files",
                        type=check_dir,
                        default=pl.Path("../batches/"),
                        required=False)
    parser.add_argument("-t", dest="ff_threshold",
                        help="first fit threshold argument",
                        type=check_float,
                        default=2.0,
                        required=False)
    parser.add_argument("-input", dest="input_file",
                        help="input_csv file",
                        type=check_file,
                        default=pl.Path("../inputs/inputs.csv"),
                        required=False)
    parser.add_argument("-w", dest="w",
                        help="weighted round robin argument",
                        type=check_int,
                        default=3,
                        required=False)
    parser.add_argument("-max", dest="mmk",
                        help="max min kernel number",
                        type=check_float,
                        default=0.75,
                        required=False)
    parser.add_argument("-min", dest="mk",
                        help="min min kernel number",
                        type=check_float,
                        default=0.4,
                        required=False)

    return parser.parse_args()


def results(result, batch):
    data = []
    times = []
    for kernel in result.queue:
        data.append(kernel.name)

    for kernel_name in batch:
        if kernel_name in data:
            times.append(result.get_kernel_time(kernel_name))
        else:
            times.append(0.0)

    name = result.name
    return pd.DataFrame([times], index=[name], columns=batch)


def export_name(input_file):
    e_n = ""
    df = pd.read_csv(input_file)
    df = df.drop(df[df.enable == 0].index)
    for index in df['name']:
        e_n += index + "_+_"

    return e_n[:-3]


def w_results(scheduler_name, device_list, batch):
    devices = []
    device_timeline = []
    for device in device_list:
        device_timeline.append(results(device, batch))
        devices.append(pd.DataFrame({f'{device.name}_time': [device.current_time],
                                    f'{device.name}_energy': [device.energy]},
                                    index=[scheduler_name]))
    df_stats = pd.concat(devices,  axis='columns')
    df_timeline = pd.concat(device_timeline)

    return df_stats, df_timeline


def w_timeline(scheduler_name, df_timeline, name, batch_name):
    timeline_dir = f"results/{name}/timeline/{batch_name[:-4]}/"
    os.makedirs(timeline_dir, exist_ok=True)
    df_timeline.to_csv(timeline_dir + scheduler_name + ".csv", sep=',', encoding='utf-8')


def get_lines(device_list):
    line = [0.0]
    for device in device_list:
        line.append(device.current_time)

    return sorted(line)


def main():
    args = get_args()
    info_file = args.info_file
    input_file = args.input_file
    batches_dir = args.batches_dir
    threshold = args.ff_threshold
    w = args.w
    mmk = args.mmk
    mk = args.mk
    batches = ["A.csv", "M+.csv", "MM.csv", "M-.csv", "S+.csv", "SS.csv", "S-.csv", "T+.csv", "TT.csv", "T-.csv"]

    e_n = export_name(input_file)

    batch_list = sc.batch_lists(batches_dir, batches)

    i = 0
    for batch in batch_list:

        # FCFS
        scheduler_name, fc = sc.scheduler_fcfs(batch, sc.dev_list(input_file, info_file))
        df_fcfs, df_fcfs_timeline = w_results(scheduler_name, fc, batch)

        w_timeline(scheduler_name, df_fcfs_timeline, e_n, batches[i])
        splt.plot_timeline(e_n, batches[i], scheduler_name, get_lines(fc))

        # Only GPU
        scheduler_name, og = sc.only_gpu(batch, sc.dev_list(input_file, info_file))
        df_only_gpu, df_only_gpu_timeline = w_results(scheduler_name, og, batch)

        w_timeline(scheduler_name, df_only_gpu_timeline, e_n, batches[i])
        splt.plot_timeline(e_n, batches[i], scheduler_name, get_lines(og))

        # Only CPU
        scheduler_name, oc = sc.only_cpu(batch, sc.dev_list(input_file, info_file))
        df_only_cpu, df_only_cpu_timeline = w_results(scheduler_name, oc, batch)

        w_timeline(scheduler_name, df_only_cpu_timeline, e_n, batches[i])
        splt.plot_timeline(e_n, batches[i], scheduler_name, get_lines(oc))

        # Round Robin
        n = 1
        scheduler_name, wrr = sc.weighted_round_robin(batch, sc.dev_list(input_file, info_file), n)
        df_wrr, df_wrr_timeline = w_results(scheduler_name, wrr, batch)

        w_timeline(scheduler_name, df_wrr_timeline, e_n, batches[i])
        splt.plot_timeline(e_n, batches[i], scheduler_name, get_lines(wrr))

        # Weighted Round Robin
        scheduler_name, wrr1 = sc.weighted_round_robin(batch, sc.dev_list(input_file, info_file), w)
        df_wrr1, df_wrr1_timeline = w_results(scheduler_name, wrr1, batch)

        w_timeline(scheduler_name, df_wrr1_timeline, e_n, batches[i])
        splt.plot_timeline(e_n, batches[i], scheduler_name, get_lines(wrr1))

        # First Fit
        scheduler_name, ff = sc.first_fit(batch, sc.dev_list(input_file, info_file), threshold)
        df_ff, df_ff_timeline = w_results(scheduler_name, ff, batch)

        w_timeline(scheduler_name, df_ff_timeline, e_n, batches[i])
        splt.plot_timeline(e_n, batches[i], scheduler_name, get_lines(ff))

        # Max Min
        max_kernel = int(len(batch) * mmk)
        scheduler_name, max_min, batch_order = sc.max_min(batch, sc.dev_list(input_file, info_file), max_kernel)
        df_max_min, df_max_min_timeline = w_results(scheduler_name, max_min, batch_order)

        w_timeline(scheduler_name, df_max_min_timeline, e_n, batches[i])
        splt.plot_timeline(e_n, batches[i], scheduler_name, get_lines(max_min))

        # Min Min
        min_kernel = int(len(batch) * mk)
        scheduler_name, mm, batch_order = sc.min_min(batch, sc.dev_list(input_file, info_file), min_kernel)
        df_mm, df_mm_timeline = w_results(scheduler_name, mm, batch_order)

        w_timeline(scheduler_name, df_mm_timeline, e_n, batches[i])
        splt.plot_timeline(e_n, batches[i], scheduler_name, get_lines(mm))

        # Export
        df_export = pd.concat([df_only_cpu, df_only_gpu, df_fcfs, df_wrr, df_wrr1, df_max_min, df_mm, df_ff])
        dir_export = f"results/{e_n}/stats/"
        os.makedirs(dir_export, exist_ok=True)
        df_export.to_csv(dir_export + batches[i], sep=',', encoding='utf-8')

        i += 1

    for batch_name in batches:
        splt.plot_time(e_n, batch_name)
        splt.plot_energy(e_n, batch_name)

    splt.all_results(e_n, batches)


if __name__ == "__main__":
    main()
