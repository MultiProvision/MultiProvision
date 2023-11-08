# MultiProvision

A Framework for multi-tenant resource provisioning in CPU-GPU environments.

## Dependencies
Before running the TRIPP framework, some packages must be installed, including:
- Pandas
- Matplotlib

## Parameters
Custom parameters for execution:
Input         | Description                   | Default
------------- | -------------                 | -------------
-b            | batches directory.            | "../batches/"
-i            | input file directory.         | "../benchmark_results/benchmark_information.csv"
-input        | Input CSV file.               | "../inputs/inputs.csv"
-h            | help                          |
-max          | max min kernel number.        | 0.75 (percentage value)
-min          | min min kernel number.        | 0.4 (percentage value)
-t            | first fit threshold argument. | 2.0
-w            | weighted round robin argument.| 3 (unitary value)

## Folder Structure
```
MultiProvision
├── batches                           # Batches files
├── benchmark_results                 # Benchmark extracted file
├── inputs                            # Input configuration file (.csv)
└── schedulers                        # Execution files and results
    ├── plots                         # Plots results
    │   ├── Architecture 1            # Folder of architecture selected
    │   |    ├── all_results          # All schedulers plots (Makespan and energy)
    │   |    ├── energy               # Energy plot
    │   |    ├── time                 # Time plot
    │   |    └── timeline             # Timeline plot
    |   ├── Architecture 2            # Folder of architecture selected
    │   |    ├──...                   # (All schedulers plots, Energy plot, Time plot, Timeline plot)
    |   └── ...                       # Folder of architecture selected...
    ├── results                       # Data results
    |    └── Architecture 1           # Folder of architecture selected
    |    |    ├── stats               # Energy and Time data
    |    |    └── timeline            # Timeline data
    |    ├── Architecture 2           # Folder of architecture selected
    │    |    ├──...                  # (Energy, Time and Timeline data)
    |    └── ...                      # Folder of architecture selected...
    ├── run_all_schedulers.py         # Execution file
    ├── scheduler_plots.py            # Plotting file 
    └── schedulers.py                 # Scheduler file
```

## Execution

Standard execution:

```
C:\MultiProvision\schedulers> python.exe run_all_schedulers.py
```

Custom parameters:

- max min 60%
- min min 30%
- weighted round robin argument 6 kernels
- first fit threshold 1.0


```
C:\MultiProvision\schedulers> python.exe run_all_schedulers.py -t 1.0 -w 6 -max 0.6 -min 0.3
```

## Important information

- Before executing a new analysis with the same architecture (same CPU-GPU) but different parameters (-max, -min, -w, and -t), it is necessary to remove all files inside the architecture folder.
