import os
import json
import pandas as pd
import matplotlib.pyplot as plt

def get_counts_from_splits(splits_folder):
    """
    Scans a folder for .json split files and counts Train/Test IDs.
    """
    all_counts = []

    # Loop through all files in the splits directory
    for filename in os.listdir(splits_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(splits_folder, filename)
            
            with open(file_path, "r") as f:
                split_data = json.load(f)
            
            train_ids = split_data.get("train_ids", [])
            test_ids = split_data.get("test_ids", [])
            
            # Extract a clean name from the filename
            # e.g., 'BRCA_random_split.json' -> 'BRCA_random'
            display_name = filename.replace("_split.json", "").replace(".json", "")

            all_counts.append({
                "Dataset": display_name,
                "Train": len(train_ids),
                "Test": len(test_ids),
                "Total": len(train_ids) + len(test_ids)
            })

    return pd.DataFrame(all_counts)


def plot_split_distributions(splits_folder, save_path=None):
    df = get_counts_from_splits(splits_folder)
    
    if df.empty:
        print("No split files found in the directory.")
        return

    # Set Dataset as index for plotting
    df = df.set_index("Dataset")

    # Plotting
    ax = df[['Train', 'Test']].plot(
        kind='bar', 
        stacked=True, 
        figsize=(12, 6), 
        color=['#3498db', '#e74c3c'], # Blue for Train, Red for Test
        alpha=0.85
    )

    plt.title("Sample Counts per Experiment Split", fontsize=15, pad=20)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.xlabel("Experiment Name (JSON Filename)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # Add text labels for the total count on top of bars
    for i, total in enumerate(df['Total']):
        ax.text(i, total + 0.5, str(total), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()