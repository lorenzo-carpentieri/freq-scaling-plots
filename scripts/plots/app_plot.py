# The script that take as input a .csv file with data time and energy of the application. 
# Given the csv only consider the row with Benchmark application.
# Generate a plot where the x-axis is the speedup while the y-axis is normalized energy.
# The speedup and normalized energy are computed considering as baseline the configuration with freq. 1600
# The speedup is computes as baseline / configuration with another freq.  The script add another column to the CSV with the computed Speedup
# The normalized energy is computed as configuration with another freq / baseline. The script add another col to the CSV with the Normalized Energy.
# The plot show the Speedup and Normalized Energy for all frequencies. The frequencies are represented by a color map of viridis color using a color bar.

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

LABEL_FONT_SIZE=14
X_Y_LABEL=12

def extract_opt_freq(df):
    # Calculate Energy Delay Product (EDP)
    df['EDP'] = df['Time Mean [ms]'] * df['Device Energy Mean [J]']

    # Find frequency that minimizes energy
    min_energy_row = df.loc[df['Device Energy Mean [J]'].idxmin()]
    min_energy_freq = min_energy_row['Core Freq [MHz]']
    min_energy_val = min_energy_row['Device Energy Mean [J]']

    # Find frequency that minimizes EDP
    min_edp_row = df.loc[df['EDP'].idxmin()]
    min_edp_freq = min_edp_row['Core Freq [MHz]']
    min_edp_val = min_edp_row['EDP']

    print(f"\nFrequency minimizing Energy: {min_energy_freq} MHz (Energy: {min_energy_val:.2f} J)")
    print(f"Frequency minimizing EDP: {min_edp_freq} MHz (EDP: {min_edp_val:.2f} ms*J)")

    return df


def print_norm_plot(app_df, app_name, default_freq, output_plot):
    # Plotting
    plt.figure(figsize=(10, 7))
    
    # Use seaborn scatterplot with hue for Core Freq and viridis colormap
    scatter = sns.scatterplot(
        data=app_df,
        x='Speedup', 
        y='Normalized Energy',
        hue='Core Freq [MHz]',
        palette='viridis', # Reversed viridis so high freq is yellow
        s=200, # Size of the points
        edgecolor='black', # Add a border to points for better visibility
        linewidth=0.5,
        zorder=2 # Ensure points are below the baseline marker
    )

    # Add a special marker for the baseline configuration (Speedup=1, Normalized Energy=1)
    default_config = plt.scatter(
        [1.0], [1.0],
        marker='X',
        color='red',
        s=250,
        edgecolor='black',
        linewidth=1,
        label=f'Baseline ({default_freq} MHz)',
        zorder=3 # Draw on top of other points
    )

    plt.title(f'{app_name} Application Performance vs. Energy (Baseline: {default_freq} MHz)')
    plt.xlabel(f'Speedup (relative to {default_freq} MHz)', fontsize=LABEL_FONT_SIZE)
    plt.ylabel(f'Normalized Energy (relative to {default_freq} MHz)', fontsize=LABEL_FONT_SIZE)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=X_Y_LABEL)
    plt.yticks(fontsize=X_Y_LABEL)
    # Add a color bar
    norm = plt.Normalize(app_df['Core Freq [MHz]'].min(), app_df['Core Freq [MHz]'].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=scatter.axes)
    cbar.set_label('Core Frequency [MHz]', fontsize=LABEL_FONT_SIZE)    
    scatter.legend_.remove() # Remove the default seaborn legend
    # Create a legend with only the specific handle for the default configuration
    plt.legend(handles=[default_config])

    plt.tight_layout()
    plot_path= output_plot.replace(".pdf", "_norm.pdf") 
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")


def print_abs_plot(app_df, app_name, default_freq, output_plot):
    # Plotting
    plt.figure(figsize=(10, 7))
    
    # Use seaborn scatterplot with hue for Core Freq and viridis colormap
    scatter = sns.scatterplot(
        data=app_df,
        x='Time Mean [ms]', 
        y='Device Energy Mean [J]',
        hue='Core Freq [MHz]',
        palette='viridis', # Reversed viridis so high freq is yellow
        s=200, # Size of the points
        edgecolor='black', # Add a border to points for better visibility
        linewidth=0.5,
        zorder=2 # Ensure points are below the baseline marker
    )
    baseline_config_abs = app_df[app_df['Core Freq [MHz]'] == default_freq]
    print(baseline_config_abs)
    default_energy = baseline_config_abs['Device Energy Mean [J]'].iloc[0]
    default_time = baseline_config_abs['Time Mean [ms]'].iloc[0]
    # Add a special marker for the baseline configuration (Speedup=1, Normalized Energy=1)
    default_config = plt.scatter(
        [default_time], [default_energy],
        marker='X',
        color='red',
        s=250,
        edgecolor='black',
        linewidth=1,
        label=f'Baseline ({default_freq} MHz)',
        zorder=3 # Draw on top of other points
    )

    plt.title(f'{app_name} Application Performance vs. Energy (Baseline: {default_freq} MHz)')
    plt.xlabel(f'Time [ms]', fontsize=LABEL_FONT_SIZE)
    plt.ylabel(f'Energy [J]', fontsize=LABEL_FONT_SIZE)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=X_Y_LABEL)
    plt.yticks(fontsize=X_Y_LABEL)
    # Add a color bar
    norm = plt.Normalize(app_df['Core Freq [MHz]'].min(), app_df['Core Freq [MHz]'].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=scatter.axes)
    cbar.set_label('Core Frequency [MHz]', fontsize=LABEL_FONT_SIZE)    
    scatter.legend_.remove() # Remove the default seaborn legend
    # Create a legend with only the specific handle for the default configuration
    plt.legend(handles=[default_config])

    plt.tight_layout()
    plot_path= output_plot.replace(".pdf", "_abs.pdf") 
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate speedup vs. normalized energy plot for miniweather application data.")
    parser.add_argument("--csv-file", type=str, required=True,
                        help="Path to the aggregated CSV file (e.g., miniweather_device.csv).")
    parser.add_argument("--baseline-freq", type=int, default=1600,
                        help="Core frequency to use as baseline for speedup and normalized energy (in MHz). Default is 1600.")
    parser.add_argument("--output-plot", type=str, default="miniweather_app_plot.png",
                        help="Path to save the output plot image.")
    parser.add_argument("--app-name", type=str, default="",
                        help="Path to save the output plot image.")
    args = parser.parse_args()
    
    app_name = args.app_name

    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file '{args.csv_file}' not found.")
        return

    df = pd.read_csv(args.csv_file)

    # Filter for application benchmark data and Rank 0
    # Assuming 'Type' column exists and 'app' refers to the entire application
    # And 'Rank' 0 typically holds the aggregated application data
    app_df = df[(df['Benchmark'] == 'application') & (df['Type'] == 'app') & (df['Rank'] == 0)].copy()

    if app_df.empty:
        print("No 'application' benchmark data found for Rank 0 in the provided CSV. Exiting.")
        return
    extract_opt_freq(app_df)
    # Ensure required columns exist
    required_cols = ['Core Freq [MHz]', 'Time Mean [ms]', 'Device Energy Mean [J]']
    if not all(col in app_df.columns for col in required_cols):
        print(f"Error: Missing one or more required columns in the CSV for plotting: {required_cols}")
        return

    # Find the baseline configuration
    baseline_config = app_df[app_df['Core Freq [MHz]'] == args.baseline_freq]

    if baseline_config.empty:
        print(f"Error: Baseline frequency {args.baseline_freq} MHz not found in the application data. Exiting.")
        return
    
    baseline_time = baseline_config['Time Mean [ms]'].iloc[0]
    baseline_energy = baseline_config['Device Energy Mean [J]'].iloc[0]

    if baseline_time == 0:
        print("Error: Baseline time is zero, cannot compute speedup. Exiting.")
        return
    if baseline_energy == 0:
        print("Warning: Baseline energy is zero. Normalized energy might be infinite or undefined.")

    # Calculate Speedup and Normalized Energy
    app_df['Speedup'] = baseline_time / app_df['Time Mean [ms]']
    app_df['Normalized Energy'] = app_df['Device Energy Mean [J]'] / baseline_energy

    print(f"\nData with calculated Speedup and Normalized Energy (baseline freq: {args.baseline_freq} MHz):")
    print(app_df[['Core Freq [MHz]', 'Time Mean [ms]', 'Device Energy Mean [J]', 'Speedup', 'Normalized Energy']].round(2))

    
    
    print_norm_plot(app_df, app_name, args.baseline_freq, args.output_plot)
    print_abs_plot(app_df, app_name, args.baseline_freq, args.output_plot)
    
    
    # plt.show() # Uncomment to display the plot immediately

if __name__ == "__main__":
    main()
