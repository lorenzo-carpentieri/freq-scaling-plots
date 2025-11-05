# Create a program that take as input the dir path to the folder containing the miniweather logs. Take the path to the dir as command line parameter using argparse
# The folder contains a number of folders named as freq-X where X is the core freq. used.
# For each folder consider the folder inside it that are named as run-Y where Y is the number representing the run
# Iterate over the file that will be named like this rank_N_device.log or rank_N_kernels.log
# Create a function for parsing the two file. 

# Create a CSV file with the follwing columns:
# Benchmark, Type,  Device Energy [J], Time [ms], Core Freq [MHz]
# Type contsins the info about the type of benchmark that can be kernel: a single kernel of miniweather, phase_total: entire phase profiled considering all the rank, phase_rank: the configuration for a single rank,
# or app: the entire application.


import argparse
import os
import re
import pandas as pd
from collections import defaultdict

# GPU energy is the energy related to a single rank
SINGLE_RANK_INFO= "GPU energy"
# All GPUs is the energy aggregated using a Reduce that take into account all the energy consumed by the devices involved in the benchmark
AGGREGATED_INFO="All GPUs energy"
# Phase name
SET_HALO="set_halo"
COMPUTE="compute"
APPLY_TEND="apply tendencies"
REDUCTIONS="reductions"
INIT="init"
# Create string to matach for collecting the info 
SET_HALO_PHASE_RANK = f"{SINGLE_RANK_INFO} {SET_HALO} [J]:" 
SET_HALO_PHASE = f"{AGGREGATED_INFO} {SET_HALO} [J]:" 
COMPUTE_PHASE_RANK = f"{SINGLE_RANK_INFO} {COMPUTE} [J]:" 
COMPUTE_PHASE = f"{AGGREGATED_INFO} {COMPUTE} [J]:" 
APPLY_TEND_PHASE_RANK = f"{SINGLE_RANK_INFO} {APPLY_TEND} [J]:" 
APPLY_TEND_PHASE = f"{AGGREGATED_INFO} {APPLY_TEND} [J]:" 
INIT_PHASE_RANK = f"{SINGLE_RANK_INFO} {INIT} [J]:" 
INIT_PHASE = f"{AGGREGATED_INFO} {INIT} [J]:" 
REDUCTIONS_PHASE_RANK = f"{SINGLE_RANK_INFO} {REDUCTIONS} [J]:" 
REDUCTIONS_PHASE = f"{AGGREGATED_INFO} {REDUCTIONS} [J]:" 
# Time info
TOTAL_ENERGY = f"{AGGREGATED_INFO} [J]: " # The energy used by entire application considering all devices
MAX_TIME= f"Max total time [s]:"
RANK_TIME= f"Rank total time [s]:"
ENERGY_RANK=f"Node name: "




def parse_device_log(filepath, rank):
    """
    Parses a single device log file and extracts relevant energy and time information.
    """
    data = {}
    with open(filepath, 'r') as f:
        content = f.read()

    def extract_values(pattern):
        """Finds all occurrences of a pattern and returns a list of float values."""
        matches = re.findall(f"^{re.escape(pattern.strip())}\\s*([\\d.]+)", content, re.MULTILINE)
        return [float(m) for m in matches]

    if int(rank)==0:
        # Application-level aggregated data
        # For these, we expect only one value, so we take the first element of the list.
        app_energy_total_list = extract_values(TOTAL_ENERGY)
        data['app_energy_total'] = app_energy_total_list[0] if app_energy_total_list else None
        app_time_max_list = extract_values(MAX_TIME)
        data['app_time_max'] = app_time_max_list[0] if app_time_max_list else None

        # For phases, multiple values can exist (one per timestep). We sum them up.
        data['app_energy_init'] = sum(extract_values(INIT_PHASE))
        data['app_energy_set_halo'] = sum(extract_values(SET_HALO_PHASE))
        data['app_energy_compute'] = sum(extract_values(COMPUTE_PHASE))
        data['app_energy_apply_tend'] = sum(extract_values(APPLY_TEND_PHASE))
        data['app_energy_reductions'] = sum(extract_values(REDUCTIONS_PHASE))

    # Per-rank data
    rank_time_list = extract_values(RANK_TIME)
    data['rank_time'] = rank_time_list[0] if rank_time_list else None
    data['rank_energy_init'] = sum(extract_values(INIT_PHASE_RANK))
    data['rank_energy_set_halo'] = sum(extract_values(SET_HALO_PHASE_RANK))
    data['rank_energy_compute'] = sum(extract_values(COMPUTE_PHASE_RANK))
    data['rank_energy_apply_tend'] = sum(extract_values(APPLY_TEND_PHASE_RANK))
    data['rank_energy_reductions'] = sum(extract_values(REDUCTIONS_PHASE_RANK))

    return data


def parse_kernels_log(filepath, rank):
    print(f"Paring kernel log: {filepath}")
    """
    Parses a single kernels log file and extracts aggregated energy and time per kernel.
    """
    kernel_data = defaultdict(lambda: {'Kernel Energy [J]': 0.0, 'Kernel Time [ms]': 0.0})
    with open(filepath, 'r') as f:
        for line in f:
            # Example line: kernel_name: init_1, kernel energy consumption [J]: 0.199091, device energy consumption per kernel [J]: 20.8182, kernel_time [ms]: 649.091
            match = re.search(
                r"kernel_name: ([\w_]+), .* device energy consumption per kernel \[J\]: ([\d.]+), kernel_time \[ms\]: ([\d.]+)",
                line
            )
            if match:
                kernel_name = match.group(1)
                energy = float(match.group(2))
                time = float(match.group(3))
                kernel_data[kernel_name]['Kernel Energy [J]'] += energy
                kernel_data[kernel_name]['Kernel Time [ms]'] += time
    return dict(kernel_data) # Convert defaultdict back to dict for cleaner return





def transform_device_data_to_rows(device_data, core_freq, run_num, rank):
    """
    Transforms the parsed data dictionary into a list of rows for the DataFrame.
    """
    rows = []
    if int(rank)==0:
        # Note: The log file provides time in seconds, so we convert to milliseconds.
        time_ms = device_data.get('app_time_max') * 1000 if device_data.get('app_time_max') is not None else None

        # Row for the entire application
        rows.append({
            'Benchmark': 'application',
            'Type': 'app',
            'Device Energy [J]': device_data.get('app_energy_total'),
            'Time [ms]': time_ms,
            'Core Freq [MHz]': core_freq,
            'Run': run_num,
            'Rank': rank
        })

        # Create a mapping from the data key to the benchmark name
        phase_mapping_all = {
            'app_energy_init': 'init',
            'app_energy_set_halo': 'set_halo',
            'app_energy_compute': 'compute',
            'app_energy_apply_tend': 'apply_tendencies',
            'app_energy_reductions': 'reductions'
        }

        # Rows for each phase
        for key, name in phase_mapping_all.items():
            if device_data.get(key) is not None:
                rows.append({
                    'Benchmark': name,
                    'Type': 'phase_all',
                    'Device Energy [J]': device_data.get(key),
                    'Time [ms]': None, # Phase-specific timings are not available in device logs
                    'Core Freq [MHz]': core_freq,
                    'Run': run_num,
                    'Rank': rank
                })
    phase_mapping_rank = {
            'rank_energy_init': 'init',
            'rank_energy_set_halo': 'set_halo',
            'rank_energy_compute': 'compute',
            'rank_energy_apply_tend': 'apply_tendencies',
            'rank_energy_reductions': 'reductions'
        }

    # Rows for each phase
    for key, name in phase_mapping_rank.items():
        if device_data.get(key) is not None:
            rows.append({
                'Benchmark': name,
                'Type': 'phase_rank',
                'Device Energy [J]': device_data.get(key),
                'Time [ms]': None, # Phase-specific timings are not available in device logs
                'Core Freq [MHz]': core_freq,
                'Run': run_num,
                'Rank': rank
            })

    return rows

def transform_kernel_data_to_rows(kernel_data, core_freq, run_num, rank):
    """
    Transforms the parsed kernel data dictionary into a list of rows for the DataFrame.
    """
    rows = []
    for kernel_name, metrics in kernel_data.items():
        rows.append({
            'Benchmark': kernel_name,
            'Type': 'kernel',
            'Kernel Energy [J]': metrics['Kernel Energy [J]'],
            'Kernel Time [ms]': metrics['Kernel Time [ms]'],
            'Core Freq [MHz]': core_freq,
            'Run': run_num,
            'Rank': rank
        })
    return rows

def aggregate_data(df, grouping_cols, agg_functions):
    """
    Groups and aggregates a DataFrame, then renames the columns.

    Args:
        df (pd.DataFrame): The DataFrame to aggregate.
        grouping_cols (list): A list of column names to group by.
        agg_functions (dict): A dictionary mapping column names to aggregation functions.

    Returns:
        pd.DataFrame: The aggregated and renamed DataFrame.
    """
    agg_df = df.groupby(grouping_cols).agg(agg_functions).reset_index()

    # Create new column names after aggregation
    new_columns = list(grouping_cols)
    for col, funcs in agg_functions.items():
        base_name = col.rsplit(' [', 1)[0]
        unit = col.rsplit(' [', 1)[1]
        for func in funcs:
            new_columns.append(f"{base_name} {func.capitalize()} [{unit}")
    agg_df.columns = new_columns
    return agg_df


def main():
    parser = argparse.ArgumentParser(description="Parse miniweather logs.")
    parser.add_argument("--log-dir", type=str, help="Path to the directory containing miniweather logs.")
    parser.add_argument("--output-file", type=str, default="miniweather_results.csv", help="Path to the output CSV file.")
    args = parser.parse_args()

    log_dir = args.log_dir
    if not os.path.isdir(log_dir):
        print(f"Error: Directory '{log_dir}' not found.")
        return
 
    device_data_rows = []
    kernel_data_rows = []
    

    for freq_folder_name in os.listdir(log_dir):
        if not freq_folder_name.startswith('freq-'): continue
        core_freq = freq_folder_name.split("-")[1] # Extract the core frequency from the folder name
        freq_path = os.path.join(log_dir, freq_folder_name)
        if not os.path.isdir(freq_path): continue

        for run_folder in os.listdir(freq_path):
            if not run_folder.startswith('run-'): continue
            run_num = run_folder.split("-")[1] # Extract the run number
            run_path = os.path.join(freq_path, run_folder)
            if not os.path.isdir(run_path): continue

            for filename in os.listdir(run_path):
                if filename.endswith(".log") and filename.startswith("rank"): # Parse only file related to rank and that are log files
                    file_path = os.path.join(run_path, filename)
                    if "device" in filename:
                        rank = re.search(r'rank_(\d+)_', filename).group(1)
                        device_data = parse_device_log(file_path, rank)
                        rows = transform_device_data_to_rows(device_data, core_freq, run_num, rank)
                        device_data_rows.extend(rows)
                    else:
                        rank = re.search(r'rank_(\d+)_', filename).group(1)
                        kernel_data = parse_kernels_log(file_path, rank)
                        # print(kernel_data)
                        rows = transform_kernel_data_to_rows(kernel_data, core_freq, run_num, rank)
                        kernel_data_rows.extend(rows)
    output_path = args.output_file

    if not device_data_rows:
        print("No data was parsed for device data. Exiting.")
    else:
        raw_device_df = pd.DataFrame(device_data_rows)     
        device_grouping_cols = ['Benchmark', 'Type', 'Core Freq [MHz]', 'Rank']
        device_agg_functions = {
            'Device Energy [J]': ['mean', 'std'],
            'Time [ms]': ['mean', 'std']
        }
        agg_device_df = aggregate_data(raw_device_df, device_grouping_cols, device_agg_functions)
        # Sort the DataFrame by 'Rank' and 'Core Freq [MHz]'
        agg_device_df = agg_device_df.sort_values(by=['Rank', 'Core Freq [MHz]'])
        device_file=output_path.replace(".csv", "_device.csv")
        agg_device_df.to_csv(device_file, index=False)
        
    if not kernel_data_rows:
        print("No data was parsed for kernels data. Exiting.")
    else:
        raw_kernel_df = pd.DataFrame(kernel_data_rows)
        
        # --- Aggregate Device Data ---

        # --- Aggregate Kernel Data ---
        kernel_grouping_cols = ['Benchmark', 'Type', 'Core Freq [MHz]', 'Rank']
        kernel_agg_functions = {
            'Kernel Energy [J]': ['mean', 'std'],
            'Kernel Time [ms]': ['mean', 'std']
        }
        agg_kernel_df = aggregate_data(raw_kernel_df, kernel_grouping_cols, kernel_agg_functions)
        kernel_file = output_path.replace(".csv", "_kernel.csv")
        agg_kernel_df = agg_kernel_df.sort_values(by=['Rank', 'Core Freq [MHz]'])
        agg_kernel_df.to_csv(kernel_file, index=False)


    print(f"Successfully parsed logs and saved data to {output_path}")
    
  
if __name__ == "__main__":
    main()
