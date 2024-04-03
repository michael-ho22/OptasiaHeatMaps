import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# List of files to process, file paths may be different for each person processing the information
file_paths = [
    r"C:\Users\mykoh\Brazos Innovation Partners\Project - Optasia - Changing Taps\HWRChar032724_RL2k_TapOut6_TapIn1_Dist3_20240327_135611_Traces.csv",
    r"C:\Users\mykoh\Brazos Innovation Partners\Project - Optasia - Changing Taps\HWRChar032724_RL2k_TapOut6_TapIn2_Dist3_20240327_140456_Traces.csv",
    r"C:\Users\mykoh\Brazos Innovation Partners\Project - Optasia - Changing Taps\HWRChar032724_RL2k_TapOut6_TapIn3_Dist3_20240327_143802_Traces.csv",
    r"C:\Users\mykoh\Brazos Innovation Partners\Project - Optasia - Changing Taps\HWRChar032724_RL2k_TapOut6_TapIn4_Dist3_20240327_144523_Traces.csv",
    r"C:\Users\mykoh\Brazos Innovation Partners\Project - Optasia - Changing Taps\HWRChar032724_RL2k_TapOut6_TapIn5_Dist3_20240327_145817_Traces.csv",
    r"C:\Users\mykoh\Brazos Innovation Partners\Project - Optasia - Changing Taps\HWRChar032724_RL2k_TapOut6_TapIn6_Dist3_20240327_150745_Traces.csv",
    r"C:\Users\mykoh\Brazos Innovation Partners\Project - Optasia - Changing Taps\HWRChar032724_RL2k_TapOut6_TapIn7_Dist3_20240327_151408_Traces.csv",
    r"C:\Users\mykoh\Brazos Innovation Partners\Project - Optasia - Changing Taps\HWRChar032724_RL2k_TapOut6_TapIn8_Dist3_20240327_151957_Traces.csv",
    r"C:\Users\mykoh\Brazos Innovation Partners\Project - Optasia - Changing Taps\HWRChar032724_RL2k_TapOut6_TapIn9_Dist3_20240327_152645_Traces.csv",
    r"C:\Users\mykoh\Brazos Innovation Partners\Project - Optasia - Changing Taps\HWRChar032724_RL2k_TapOut6_TapIn10_Dist3_20240327_153233_Traces.csv"
]

# Initialize a DataFrame to hold aggregated data
aggregated_data = pd.DataFrame()

# Process each file
for i, file_path in enumerate(file_paths, start=1):
    df = pd.read_csv(file_path)

    # Creating a summary by frequency
    if 'Channel of 2 Magnitude (dB)' in df.columns:
        temp_df = df[['Channel of 2 Magnitude (dB)']].copy()
        temp_df['TapIn'] = i  # Record the TapIn number for this batch
        aggregated_data = pd.concat([aggregated_data, temp_df])

# Display the first few rows of the dataframe to understand its structure
aggregated_data.head()

df_example = pd.read_csv(file_paths[0])

df_example.head()

# Because the first couple of lines are irrelevant, skip the initial metadata rows and try to find the actual data

# Attempt to load the file again, skipping the first 6 initial rows
df_example_adjusted = pd.read_csv(file_paths[0], skiprows=6)

# Just checking to see if data seems good here
df_example_adjusted.head()

# Initialize a DataFrame to hold the required data for the heatmap
data_for_heatmap = pd.DataFrame()

# Process each file extracting the necessary columns
for tapin, file_path in enumerate(file_paths, start=1):
    df = pd.read_csv(file_path, skiprows=6)  # Adjusted skiprows based on the structure seen above
    temp_df = df[['% Frequency (Hz)', ' Channel 2 Magnitude (dB)']].copy()
    temp_df['TapIn'] = tapin
    data_for_heatmap = pd.concat([data_for_heatmap, temp_df])


# First, let's check if frequencies are unique within each TapIn or if we need to aggregate further
data_for_heatmap.head()

# Pivot the DataFrame to structure it correctly for the heatmap
# We want the index to be the frequencies, columns to be TapIns, and values to be the magnitude in dB
heatmap_data = data_for_heatmap.pivot_table(index='% Frequency (Hz)', columns='TapIn', values=' Channel 2 Magnitude (dB)')


plt.figure(figsize=(15, 10))  # Adjust size as needed
ax = sns.heatmap(heatmap_data, cmap='viridis', cbar_kws={'label': 'Intensity (dB)'}, vmin=-80, vmax=0)
plt.title('Heatmap of Intensity (dB) by TapIn - TapOut 6', fontweight='bold', fontsize=20)
plt.xlabel('TapIn', fontweight='bold')
plt.ylabel('Frequency (Hz)', fontweight='bold')

# Determine the number of desired ticks (e.g., 20 ticks for this example)
num_ticks = 20

# Calculate the positions for the desired number of ticks
tick_positions = np.linspace(start=0, stop=len(heatmap_data.index) - 1, num=num_ticks, dtype=int)

# Select the labels for the calculated positions
tick_labels = [heatmap_data.index[i] for i in tick_positions]

# Set the custom ticks and labels
ax.set_yticks(tick_positions)
ax.set_yticklabels(tick_labels, rotation=0)

plt.show()



# # Calculate bin edges - aiming for a reasonable number of bins across the frequency range
# min_freq = data_for_heatmap['% Frequency (Hz)'].min()
# max_freq = data_for_heatmap['% Frequency (Hz)'].max()
# print(min_freq)
# print(max_freq)
# num_bins = 50  # Adjust the number of bins as necessary for clarity and resolution

# # Generate bin edges from min to max frequency
# bin_edges = np.linspace(min_freq, max_freq, num_bins)

# # Bin frequencies and create a new column for the binned frequency
# data_for_heatmap['Frequency Bin'] = pd.cut(data_for_heatmap['% Frequency (Hz)'], bins=bin_edges, labels=np.arange(1, num_bins), include_lowest=True)

# # Pivot the DataFrame for the heatmap, using the binned frequency this time
# heatmap_data_binned = data_for_heatmap.pivot_table(index='Frequency Bin', columns='TapIn', values=' Channel 2 Magnitude (dB)', aggfunc=np.mean)

# # Create the heatmap with binned frequencies
# plt.figure(figsize=(12, 8))
# sns.heatmap(heatmap_data_binned, cmap='viridis', cbar_kws={'label': 'Intensity (dB)'})
# plt.title('Heatmap of Intensity (dB) by TapIn (with Binned Frequencies)', fontweight='bold')
# plt.xlabel('TapIn', fontweight='bold')
# plt.ylabel('Frequency (Hz)', fontweight='bold')
# plt.xticks(np.arange(len(heatmap_data_binned.columns)) + 0.5, heatmap_data_binned.columns)  # Adjust the x-ticks to center
# note = "Note: Each bin range is ~398,000 Hz"
# plt.text(x=0, y=-5, s=note, fontsize=12, color='red')
# plt.show()
