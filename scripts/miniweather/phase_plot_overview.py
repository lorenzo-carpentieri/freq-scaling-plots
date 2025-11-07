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

MW_PHASES=["application", "set_halo", "compute", "apply_tendencies"]
def add_speedup_and_norm_energy(df, default_freq):
    # Ensure required columns exist
    required_cols = ['Core Freq [MHz]', 'Time Mean [ms]', 'Device Energy Mean [J]']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing one or more required columns in the CSV for plotting: {required_cols}")
        return

    # Find the baseline configuration
    baseline_config = df[df['Core Freq [MHz]'] == default_freq]

    if baseline_config.empty:
        print(f"Error: Baseline frequency {default_freq} MHz not found in the application data. Exiting.")
        return
    
    baseline_time = baseline_config['Time Mean [ms]'].iloc[0]
    baseline_energy = baseline_config['Device Energy Mean [J]'].iloc[0]

    if baseline_time == 0:
        print("Error: Baseline time is zero, cannot compute speedup. Exiting.")
        return
    if baseline_energy == 0:
        print("Warning: Baseline energy is zero. Normalized energy might be infinite or undefined.")

    # Calculate Speedup and Normalized Energy
    df['Speedup'] = baseline_time / df['Time Mean [ms]']
    df['Normalized Energy'] = df['Device Energy Mean [J]'] / baseline_energy

    print(f"\nData with calculated Speedup and Normalized Energy (baseline freq: {default_freq} MHz):")
    print(df[['Core Freq [MHz]', 'Time Mean [ms]', 'Device Energy Mean [J]', 'Speedup', 'Normalized Energy']].round(2))
    return df



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


def print_norm_plot(app_df, phase_df, app_name, default_freq, output_plot):
    
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, len(MW_PHASES), figsize=(20, 5), sharey=True)
    if len(MW_PHASES) == 1:
        axes = [axes] # Ensure axes is iterable even for a single subplot

    # Combine all frequency data to get global min/max for the color bar
    all_freqs = pd.concat([app_df['Core Freq [MHz]'], phase_df['Core Freq [MHz]']]).unique()
    norm = plt.Normalize(all_freqs.min(), all_freqs.max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    for i, phase_name in enumerate(MW_PHASES):
        ax = axes[i]
        
        # Select the data framw that we want to plot
        current_df = None
        if phase_name == "application":
            current_df = app_df.copy()
        else:
            current_df = phase_df[phase_df['Benchmark'] == phase_name].copy()
        
        # Check if the selelcted data frame is empty
        if current_df.empty:
            print(f"No data for phase: {phase_name}. Skipping plot for this phase.")
            ax.set_title(f"{phase_name} (No Data)")
            ax.set_xlabel('Speedup')
            ax.set_ylabel('Normalized Energy')
            continue

        # Calculate Speedup and Normalized Energy for the current phase.
        # Now we can compute the speedup and norm energy for the specific phase
        current_df = add_speedup_and_norm_energy(current_df, default_freq)
        
        if current_df is None: continue
        # Use seaborn scatterplot with hue for Core Freq and viridis colormap
        scatter = sns.scatterplot(
            data=current_df,
            x='Speedup', 
            y='Normalized Energy',
            hue='Core Freq [MHz]',
            palette='viridis',
            s=80,
            # edgecolor='black',
            linewidth=0.5,
            zorder=2,
            ax=ax,
            legend=False # Disable individual legends
        )

        # Add a special marker for the baseline configuration (Speedup=1, Normalized Energy=1)
        default_config = ax.scatter(
            [1.0], [1.0],
            marker='X',
            color='black',
            s=60,
            # edgecolor='black',
            linewidth=1,
            label=f'Default Configuraiton ({default_freq} MHz)',
            zorder=3
        )

        # Find and annotate the point with minimum energy
        min_energy_row = current_df.loc[current_df['Device Energy Mean [J]'].idxmin()]
        min_energy_freq = min_energy_row['Core Freq [MHz]']
        min_energy_speedup = min_energy_row['Speedup']
        min_energy_norm_energy = min_energy_row['Normalized Energy']
        min_energy = ax.scatter(
            min_energy_row['Speedup'], min_energy_row['Normalized Energy'],
            marker='X',
            color='orange',
            s=60,
            # edgecolor='black',
            linewidth=0.5,
            label=f'Min. Energy Freq ({min_energy_freq} MHz)',
            zorder=3
        )

        # ax.annotate(
        #     f'Min Energy: {min_energy_freq} MHz',
        #     xy=(min_energy_speedup, min_energy_norm_energy),
        #     xytext=(min_energy_speedup-0.03, min_energy_norm_energy + 0.2), # Position text slightly above the point, adjust as needed
        #     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
        #     fontsize=10,
        #     ha='center', # Horizontal alignment
        #     va='bottom', # Vertical alignment
        #     bbox=dict(boxstyle="round,pad=0.3", fc="green", ec="black", lw=1, alpha=0.8)
        # )


        ax.set_title(f'{phase_name.replace("_", " ").title()}')
        ax.set_xlabel(f'Speedup (vs. {default_freq} MHz)', fontsize=X_Y_LABEL)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', which='major', labelsize=X_Y_LABEL)
        ax.legend(handles=[default_config, min_energy])
        
    # Set common Y label
    axes[0].set_ylabel(f'Normalized Energy (vs. {default_freq} MHz)', fontsize=LABEL_FONT_SIZE)

    # fig.subplots_adjust(right=0.85) 
    cbar = fig.colorbar(sm, ax=axes[len(MW_PHASES)-1])
    cbar.set_label('Core Frequency [MHz]', fontsize=LABEL_FONT_SIZE)

    # Add a title for the entire figure
    fig.suptitle(f'{app_name} Performance vs. Energy for Phases (Baseline: {default_freq} MHz)', fontsize=16)

    plt.tight_layout() # Adjust layout to prevent title overlap and fit colorbar
    plt.savefig(output_plot)
    print(f"\nPlot saved to {output_plot}")




def main():
    parser = argparse.ArgumentParser(description="Generate speedup vs. normalized energy plot for miniweather application data.")
    parser.add_argument("--csv-file-app", type=str, required=True,
                        help="Path to the aggregated CSV file (e.g., miniweather_device.csv).")
    parser.add_argument("--csv-file-phase", type=str, required=True,
                        help="Path to the aggregated CSV file with the phase info (e.g., miniweather_phase.csv).")
    parser.add_argument("--baseline-freq", type=int, default=1600,
                        help="Core frequency to use as baseline for speedup and normalized energy (in MHz). Default is 1600.")
    
    parser.add_argument("--output-plot", type=str, default="miniweather_phases_overview.pdf",
                        help="Path to save the output plot image.")
    args = parser.parse_args()

    if not os.path.exists(args.csv_file_app):
        print(f"Error: CSV file '{args.csv_file_app}' not found.")
        return
    if not os.path.exists(args.csv_file_phase):
        print(f"Error: CSV file '{args.csv_file_phase}' not found.")
        return
    
    df_app = pd.read_csv(args.csv_file_app)
    df_phase = pd.read_csv(args.csv_file_phase)
    
    
    
      
    # Filter for the specified phase and Rank 0
    # Assuming 'Type' column exists and 'phase_all' refers to the phase info aggregated with SUM for energy and MAX for time
    df_phase = df_phase[(df_phase['Type'] == 'phase_all') & (df_phase['Rank'] == 0)].copy() 
    # Extract the results for the entire application
    df_app = df_app[(df_app['Benchmark'] == 'application') & (df_app['Type'] == 'app') & (df_app['Rank'] == 0)].copy()

    
    if df_app.empty:
        print("No 'application' benchmark data found for Rank 0 in the provided CSV. Exiting.")
        return
    
    if df_phase.empty:
        print("No 'phase_all' benchmark data found for Rank 0 in the provided CSV. Exiting.")
        return

    # Compute speedup and normalized energy
    # df_app = add_speedup_and_norm_energy(df_app, args.baseline_freq)
    # df_phase = add_speedup_and_norm_energy(df_phase, args.baseline_freq)
    
    
    print_norm_plot(df_app, df_phase, "Miniweather" , args.baseline_freq, args.output_plot)

if __name__ == "__main__":
    main()
