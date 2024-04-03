''' Michael Ho - 4/3/24 '''
''' This program will produce all heat maps at once '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Specify the path to your folder, this will need to be changed accordingly to file path in your won device!!
folder_path = r'C:\Users\mykoh\OneDrive\Documents\BIP\Optasia\Project - Optasia - Changing Taps'

# List all files in the folder
files = os.listdir(folder_path)
f=open('files.txt', 'w+')

# Filter out every other file
every_other_file = files[1::2]  # This slices the list to get every second element starting from index 1

# Create full paths to the files
file_paths = [os.path.join(folder_path, filename) for filename in every_other_file]


def process_file_chunk(chunk_file_paths, chunk_index):
    # Initialize a DataFrame to hold the required data for the heatmap
    data_for_heatmap = pd.DataFrame()

    # Process each file extracting the necessary columns
    for tapin, file_path in enumerate(chunk_file_paths, start=1):
        df = pd.read_csv(file_path, skiprows=6) # The first six rows are irrelevant, so we can skip em
        temp_df = df[['% Frequency (Hz)', ' Channel 2 Magnitude (dB)']].copy()
        temp_df['TapIn'] = tapin  # Use tapin directly without adjustment
        data_for_heatmap = pd.concat([data_for_heatmap, temp_df])

    # Pivot the DataFrame to structure it correctly for the heatmap
    heatmap_data = data_for_heatmap.pivot_table(index='% Frequency (Hz)', columns='TapIn', values=' Channel 2 Magnitude (dB)')

    # Plotting
    plt.figure(figsize=(15, 10))
    ax = sns.heatmap(heatmap_data, cmap='viridis', cbar_kws={'label': 'Intensity (dB)'}, vmin=-80, vmax=0)
    plt.title(f'Heatmap of Intensity (dB) by TapIn - TapOut {chunk_index + 1}', fontweight='bold', fontsize=20)
    plt.xlabel('TapIn', fontweight='bold')
    plt.ylabel('Frequency (Hz)', fontweight='bold')

    # Determine the number of desired ticks (20 ticks for this example)
    num_ticks = 20
    
    # Calculate the positions for the desired number of ticks
    tick_positions = np.linspace(start=0, stop=len(heatmap_data.index) - 1, num=num_ticks, dtype=int)

    # Select the labels for the calculated positions
    tick_labels = [heatmap_data.index[i] for i in tick_positions]

    # Set the custom ticks and labels
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, rotation=0)

    plt.show()

# Divide file_paths into chunks of 10 and process each chunk
chunk_size = 10
file_chunks = [file_paths[i:i + chunk_size] for i in range(0, len(file_paths), chunk_size)]

for chunk_index, file_chunk in enumerate(file_chunks):
    process_file_chunk(file_chunk, chunk_index)