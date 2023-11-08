import pandas as pd
from pathlib import Path


class Device:
    def __init__(self, name, hardware):
        self.name = name
        self.hardware = hardware
        self.kernel_list = []
        self.current_time = 0.0
        self.energy = 0.0
        self.queue = []

    def get_name(self):
        return self.name

    def get_current_time(self):
        return self.current_time

    def kernel_list_data(self, filename, device_type, device_name):
        self.kernel_list = read_device_bench_csv(filename, device_type, device_name)

    def get_kernel_time(self, kernel_name):
        for kernel in self.kernel_list:
            if kernel.get_name() == kernel_name:
                return kernel.get_time()

    def get_kernel_by_name(self, name):
        for kernel in self.kernel_list:
            if kernel.name == name:
                return kernel

    def enqueue(self, kernel_name):
        self.queue.append(self.get_kernel_by_name(kernel_name))
        self.current_time += self.get_kernel_by_name(kernel_name).time
        self.energy += self.get_kernel_by_name(kernel_name).energy


class Kernel:
    def __init__(self, name, time, energy):
        self.name = name
        self.time = time
        self.energy = energy

    def get_name(self):
        return self.name

    def get_time(self):
        return self.time


def lower_current_time(dev_arr, kernel, device_type):
    list1 = []
    list2 = []

    if device_type == "normal":
        for device in dev_arr:
            list1.append([device.name, device.get_current_time(), device.get_kernel_time(kernel)])
    else:
        for device in dev_arr:
            if device.hardware == device_type:
                list1.append([device.name, device.get_current_time(), device.get_kernel_time(kernel)])

    list1.sort(key=lambda x: x[1])
    low = list1[0][1]
    for cur in list1:
        if cur[1] == low:
            list2.append(cur)
    list2.sort(key=lambda a: a[2])
    name = list2[0][0]

    for device in dev_arr:
        if device.name == name:
            name = device
    index = name
    return dev_arr.index(index)


def speedup(device_list, kernel):
    cpu_index = lower_current_time(device_list, kernel, "cpu")
    cpu_time = device_list[cpu_index].get_kernel_time(kernel)
    gpu_index = lower_current_time(device_list, kernel, "gpu")
    gpu_time = device_list[gpu_index].get_kernel_time(kernel)
    spd = cpu_time / gpu_time
    return spd, cpu_index, gpu_index


def sort_by_speedup(dev_arr1, batch, order):
    speedup_list = []
    kernel_list = []
    for kernel in batch:
        cpu_index = lower_current_time(dev_arr1, kernel, "cpu")
        cpu_time = dev_arr1[cpu_index].get_kernel_time(kernel)
        gpu_index = lower_current_time(dev_arr1, kernel, "gpu")
        gpu_time = dev_arr1[gpu_index].get_kernel_time(kernel)
        spd = cpu_time / gpu_time
        speedup_list.append([kernel, spd])

    speedup_list.sort(reverse=order, key=lambda a: a[1])

    for kernel in speedup_list:
        kernel_list.append(kernel[0])

    return kernel_list


# ----------- SCHEDULERS ---------------
def scheduler_fcfs(batch, dev_arr2):
    for kernel in batch:
        kernel_name = kernel
        cpu_time = lower_current_time(dev_arr2, kernel, "cpu")
        gpu_time = lower_current_time(dev_arr2, kernel, "gpu")
        if dev_arr2[cpu_time].current_time == dev_arr2[gpu_time].current_time:
            dev_arr2[gpu_time].enqueue(kernel_name)
        elif dev_arr2[cpu_time].current_time < dev_arr2[gpu_time].current_time:
            dev_arr2[cpu_time].enqueue(kernel_name)
        else:
            dev_arr2[gpu_time].enqueue(kernel_name)

    return "FCFS", dev_arr2


def weighted_round_robin(batch, device_list, n=1):
    if n < 1:
        raise ValueError("RoundRobin: n must be greater than 1!")
    for i in range(len(batch)):
        if i % (n+1) == 0:
            index = lower_current_time(device_list, batch[i], "cpu")
            device_list[index].enqueue(batch[i])
        else:
            index = lower_current_time(device_list, batch[i], "gpu")
            device_list[index].enqueue(batch[i])

    return f"wrr({n})", device_list


def first_fit(batch, dev_arr3, gpu_speed_th=2.0):
    for kernel in batch:
        spd, cpu_index, gpu_index = speedup(dev_arr3, kernel)
        if spd > gpu_speed_th:
            dev_arr3[gpu_index].enqueue(kernel)
        else:
            dev_arr3[cpu_index].enqueue(kernel)

    return "First Fit", dev_arr3


def max_min(batch, device_list, max_kernels_on_gpu=5):
    batch = sort_by_speedup(device_list, batch, True)
    batch_gpu = batch[:max_kernels_on_gpu]
    batch_cpu = batch[max_kernels_on_gpu:]
    for kernel in batch_gpu:
        index = lower_current_time(device_list, kernel, "gpu")
        device_list[index].enqueue(kernel)
    for kernel in batch_cpu:
        index = lower_current_time(device_list, kernel, "cpu")
        device_list[index].enqueue(kernel)

    return f"Max Min({max_kernels_on_gpu})", device_list, batch


def min_min(batch, device_list, max_kernels_on_gpu=5):
    batch = sort_by_speedup(device_list, batch, False)
    batch_cpu = batch[:max_kernels_on_gpu]
    batch_gpu = batch[max_kernels_on_gpu:]
    for kernel in batch_gpu:
        index = lower_current_time(device_list, kernel, "gpu")
        device_list[index].enqueue(kernel)
    for kernel in batch_cpu:
        index = lower_current_time(device_list, kernel, "cpu")
        device_list[index].enqueue(kernel)

    return f"Min Min({max_kernels_on_gpu})", device_list, batch


# ------------- SINGLE DEVICE EXECUTION ----------------
def only_gpu(batch, device_list):
    for kernel in batch:
        index = lower_current_time(device_list, kernel, "gpu")
        device_list[index].enqueue(kernel)
    return "Only Gpu", device_list


def only_cpu(batch, dev_arr4):
    for kernel in batch:
        index = lower_current_time(dev_arr4, kernel, "cpu")
        dev_arr4[index].enqueue(kernel)
    return "Only Cpu", dev_arr4


# ------ BATCH AND CSV FILE METHODS -----------
def read_device_bench_csv(filename, device, name):
    benchmark_data = []
    df1 = pd.read_csv(filename, header=[0, 1], index_col=0, skipinitialspace=True)
    df1 = df1.dropna(axis=0, how="any")

    if device == "cpu":
        if name not in df1:
            raise ValueError(f"cpu device {device} not found in file {filename}")
    if device == "gpu":
        if name not in df1:
            raise ValueError(f"gpu device {device} not found in file {filename}")

    for index in df1.index:
        if device == "gpu":
            benchmark_data.append(Kernel(filter_benchmark_name(index),
                                         df1[name]["time (s)"][index], df1[name]["energy (J)"][index]))
        if device == "cpu":
            benchmark_data.append(Kernel(filter_benchmark_name(index),
                                         df1[name]["time (s)"][index], df1[name]["pkg energy (J)"][index]))

    return benchmark_data


def dev_list(input_file, input_info):
    df = pd.read_csv(input_file)
    df = df.drop(df[df.enable == 0].index)
    dev_arr5 = []
    for index, row in df.iterrows():
        for i in range(int(row['number'])):
            device = Device(row['name']+"_"+str(i), row['type'])
            device.kernel_list_data(input_info, row['type'], row['name'])
            dev_arr5.append(device)

    return dev_arr5


def batch_lists(batch_dir, batches):
    kernel_list = []
    batch_list = []
    for batch in batches:
        df = pd.read_csv(str(Path(batch_dir))+"/"+batch, header=None)
        for index, row in df.iterrows():
            kernel_list.append(filter_benchmark_name(row[0]))
        batch_list.append(kernel_list)
        kernel_list = []
    return batch_list


def filter_benchmark_name(bench_name):
    return bench_name.strip().lower()
