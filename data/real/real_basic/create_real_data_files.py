# setup
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def scatter(a, b, log=False, log10=False, diag=True, ma=0, ax=None, c='black', s=1,
            label=None, alpha=1, aspect=False, cmap=None, return_scat_obj=False, args={}):
    # Scatters a vs b of ax (default plt). Can apply:
    #   -   log(x+1) transformation (optional)
    #   -   label (optional)
    #   -   color (default black)
    #   -   size of point (default 1)
    #   -   alpha factor (default 1)
    #   -   aspect ratio equal (optional)
    #   -   diagonal line from 0 to ma (if provided) or to max(a,b) (optional)

    ax_ = ax if ax is not None else plt
    if log:
        a = a+1
        b = b+1
        ax_.semilogx(base=2)
        ax_.semilogy(base=2)
    elif log10:
        a = a+1
        b = b+1
        ax_.semilogx(base=10)
        ax_.semilogy(base=10)
    if diag:
        if ma == 0: ma = np.max([a.max(), b.max()])
        ax_.plot([0, ma], [0, ma], linewidth=1, c='black')
    if type(c) is str or type(c) == np.ndarray:
        scat = ax_.scatter(a, b, s=s, c=c, alpha=alpha, label=label, cmap=cmap, **args)
    else:
        scat = ax_.scatter(a, b, s=s, color=c, alpha=alpha, label=label, cmap=cmap, **args)
    if aspect:
        if ax is None: ax_.gca().set_aspect('equal')
        else: ax_.set_aspect('equal')
    if return_scat_obj: return ax_, scat
    else: return ax_



def add_labels(title=None, xlabel=None, ylabel=None, xlim=None, ylim=None,
               xticks=None, xticklabels=None, xtickrotation=None,
               yticks=None, yticklabels=None, ytickrotation=None,
               aspect=False, legend=False, ax=None):
    # Applies labels (if provided)

    if ax is None or ax is plt:
        if title is not None: plt.title(title)
        if xlabel is not None: plt.xlabel(xlabel)
        if ylabel is not None: plt.ylabel(ylabel)
        if xlim is not None: plt.xlim(xlim)
        if ylim is not None: plt.ylim(ylim)
        if xticks is not None:
            if xtickrotation is not None: plt.xticks(xticks, xticklabels, rotation=xtickrotation)
            else: plt.xticks(xticks, xticklabels)
        if yticks is not None:
            if ytickrotation is not None: plt.yticks(yticks, yticklabels, rotation=ytickrotation)
            else: plt.yticks(yticks, yticklabels)
        if aspect: plt.gca().set_aspect('equal')
        if legend: plt.legend()
        return plt
    else:
        if title is not None: ax.set_title(title)
        if xlabel is not None: ax.set_xlabel(xlabel)
        if ylabel is not None: ax.set_ylabel(ylabel)
        if xlim is not None: ax.set_xlim(xlim)
        if ylim is not None: ax.set_ylim(ylim)
        if xticks is not None: ax.set_xticks(xticks)
        if yticks is not None: ax.set_yticks(yticks)
        if xticklabels is not None:
            if xtickrotation is not None: ax.set_xticklabels(xticklabels, rotation=xtickrotation)
            else: ax.set_xticklabels(xticklabels)
        if yticklabels is not None:
            if ytickrotation is not None: ax.set_yticklabels(yticklabels, rotation=ytickrotation)
            else: ax.set_yticklabels(yticklabels)
        if aspect: ax.set_aspect('equal')
        if legend: ax.legend()
        return ax


def show(title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, aspect=False,
         fig=plt, figname=None, tight_layout=True, dpi=300,
         savefig=True, showfig=True, legend=False, savepdf=False):
    # Finishes plot:
    #   -   Applies labels (if provided)
    #   -   Applies legend and tight_layout (optional)
    #   -   Saves to figname (optional)
    #   -   Shows fig (optional - otherwise delete it)

    if type(fig) is plt.Figure and title is not None: fig.suptitle(title)
    if type(fig) is not plt.Figure:
        add_labels(title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, aspect=aspect, ax=None)
    else:
        add_labels(xlabel=xlabel, ylabel=ylabel, aspect=aspect, ax=None)

    if legend: plt.legend()
    if tight_layout: fig.tight_layout()
    if savefig and figname is not None: fig.savefig(figname.parent/f'{figname.name}.png', dpi=dpi)
    if savepdf and figname is not None: fig.savefig(figname.parent/f'{figname.name}.pdf')
    if showfig: plt.show()
    else: fig.clf()
# Loading data
### Resolving Path issue
from pathlib import Path
import sys
# samples files
script_dir = Path(__file__).resolve().parent
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
# samples files
# healthy_path = Path('../../data/real/GeneMatrix_H3K4me3_healthy.csv')
# diseaseA_path = Path('../../data/real/GeneMatrix_H3K4me3_crc.csv')
# diseaseB_path = Path('../../data/real/GeneMatrix_H3K4me3_sclc.csv')

# # thetas files
# theta_A_path = Path('../../data/real/theta_CRC_passedQC.csv')
# theta_B_path = Path('../../data/real/SCLC_theta.csv')
def clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures each patient is represented only once by randomly 
    selecting one sample per patient ID (prefix before '_').
    """
    patient_ids = df.index.to_series().apply(lambda x: x.split('_')[0])
    return df.loc[patient_ids.groupby(patient_ids).apply(lambda g: g.index[0])]

healthy_path = (script_dir / '../../real/GeneMatrix_H3K4me3_healthy.csv').resolve()
diseaseA_path = (script_dir / '../../real/GeneMatrix_H3K4me3_crc.csv').resolve()
diseaseB_path = (script_dir / '../../real/GeneMatrix_H3K4me3_sclc.csv').resolve()

# thetas files
theta_A_path = (script_dir / '../../real/theta_CRC_passedQC.csv').resolve()
theta_B_path = (script_dir / '../../real/SCLC_theta.csv').resolve()

print(f'{healthy_path}')
### Load read data
import pandas as pd

df_real_healthy = pd.read_csv(healthy_path, index_col=0)
df_real_cancerA = pd.read_csv(diseaseA_path, index_col=0)
df_real_cancerB = pd.read_csv(diseaseB_path, index_col=0)

df_clean_healthy = clean_rows(df_real_healthy.T).T
df_clean_cancerA = clean_rows(df_real_cancerA.T).T
df_clean_cancerB = clean_rows(df_real_cancerB.T).T

print(f"Healthy samples before: {df_real_healthy.shape[1]}, after cleaning: {df_clean_healthy.shape[1]}")
print(f"Cancer A samples before: {df_real_cancerA.shape[1]}, after cleaning: {df_clean_cancerA.shape[1]}")
print(f"Cancer B samples before: {df_real_cancerB.shape[1]}, after cleaning: {df_clean_cancerB.shape[1]}")

df_combined_disease = pd.concat([df_clean_cancerA, df_clean_cancerB], axis=1)

print(f"Shape of combined disease file: {df_combined_disease.shape}")
valid_samples_A = df_clean_cancerA.columns
valid_samples_B = df_clean_cancerB.columns

metadata_A = pd.read_csv(theta_A_path)
metadata_B = pd.read_csv(theta_B_path)

dropped_samples_A = set(df_real_cancerA.columns) - set(df_clean_cancerA.columns)
dropped_samples_B = set(df_real_cancerB.columns) - set(df_clean_cancerB.columns)

# Filter the metadata DataFrames using .isin()
# This ensures we only keep the theta values for samples that still exist in our clean matrices
metadata_A = metadata_A[metadata_A['Unnamed: 0'].isin(valid_samples_A)].copy()
metadata_B = metadata_B[metadata_B['Unnamed: 0'].isin(valid_samples_B)].copy()

# Reset the index of the metadata so it's clean (optional but good practice)
metadata_A.reset_index(drop=True, inplace=True)
metadata_B.reset_index(drop=True, inplace=True)


# Rename the column back to something sensible for clarity
metadata_A.rename(columns={'Unnamed: 0': 'Sample_ID'}, inplace=True)
metadata_B.rename(columns={'Unnamed: 0': 'Sample_ID'}, inplace=True)
print(f"Metadata A aligned. Remaining thetas: {len(metadata_A)}")
print(f"Metadata B aligned. Remaining thetas: {len(metadata_B)}")

df_combined_theta = pd.concat([metadata_A, metadata_B], axis=0)
print(f"Shape of combined theta file: {df_combined_theta.shape}")
disease_samples = list(df_combined_disease.columns)
theta_samples = list(df_combined_theta['Sample_ID'])
if disease_samples == theta_samples:
    print("✅ Alignment check passed! Thetas and disease samples match perfectly.")
    # Set the index so to_csv writes it correctly even if no realignment was needed
    df_combined_theta = df_combined_theta.set_index('Sample_ID')
else:
    print("⚠️ Misalignment detected! Re-aligning thetas to match disease columns...")
    # Force the theta dataframe to reorder itself to match the exact column order 
    # CRITICAL: Do NOT use .reset_index() here. The string sample names MUST be the index.
    df_combined_theta = df_combined_theta.set_index('Sample_ID').reindex(disease_samples)

# Update the check to look at .index instead of the 'Sample_ID' column
updated_theta_samples = list(df_combined_theta.index)
print(f"Post-Alignment Check - Do the lists match exactly? {disease_samples == updated_theta_samples}")
n_disease_samples = df_combined_disease.shape[1]

# Randomly sample columns from the healthy dataframe with replacement (axis=1)
# replace=True allows us to draw more samples than the healthy file actually contains
df_healthy_resampled = df_real_healthy.sample(n=n_disease_samples, replace=True, axis=1)
print(f"Resampled healthy data to shape: {df_healthy_resampled.shape}")

# Extract values into numpy arrays for fast, index-free math
mix_vals = df_combined_disease.values
healthy_vals = df_healthy_resampled.values

# Extract thetas and reshape from (N,) to (1, N) so it broadcasts across the columns
thetas = df_combined_theta['data_list'].values
thetas_2d = thetas.reshape(1, -1)

# Apply the formula: Pure = (Mix - (1 - Theta) * Healthy) / Theta
pure_disease_vals = (mix_vals - (1 - thetas_2d) * healthy_vals) / thetas_2d

# Clip at 0 (gene expression cannot be negative, and dividing by small thetas can cause wild swings)
pure_disease_vals = np.clip(pure_disease_vals, a_min=0, a_max=None)

# Cast back into a Pandas DataFrame using the original genes (index) and sample names (columns)
df_pure_combined = pd.DataFrame(
    pure_disease_vals,
    index=df_combined_disease.index,
    columns=df_combined_disease.columns
)

print(f"Calculated pure disease profiles. Shape: {df_pure_combined.shape}")

# 0 = Healthy, 1 = Disease A (CRC), 2 = Disease B (SCLC)
disease_labels = [1] * df_clean_cancerA.shape[1] + [2] * df_clean_cancerB.shape[1]

df_clean_healthy.loc['disease_type'] = 0
df_combined_disease.loc['disease_type'] = disease_labels
df_pure_combined.loc['disease_type'] = disease_labels

df_clean_healthy.to_csv(script_dir / 'healthy_data.csv')
df_combined_disease.to_csv(script_dir / 'disease_data.csv')
df_combined_theta.to_csv(script_dir / 'theta_values.csv')
df_pure_combined.to_csv(script_dir / 'pure_disease_truth.csv')

print(f"All datasets saved successfully to: {script_dir}")

import matplotlib.pyplot as plt
import numpy as np

# 1. Gather the counts from your existing DataFrames
# Format: (Label, Before Count, After Count)
sample_data = [
    ('Healthy', df_real_healthy.shape[1], df_clean_healthy.shape[1]),
    ('Disease A (CRC)', df_real_cancerA.shape[1], df_clean_cancerA.shape[1]),
    ('Disease B (SCLC)', df_real_cancerB.shape[1], df_clean_cancerB.shape[1])
]

# Sort the data based on the 'After Cleaning' counts (Descending order)
sample_data.sort(key=lambda x: x[2], reverse=True)

# Extract into lists for plotting
labels = [item[0] for item in sample_data]
before_counts = [item[1] for item in sample_data]
after_counts = [item[2] for item in sample_data]

# 2. Setup the bar chart positioning
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

# 3. Create the plot
fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, before_counts, width, label='Before Cleaning', color='#d43220') # Red
rects2 = ax.bar(x + width/2, after_counts, width, label='After Cleaning', color='#2ecc71')  # Green

# 4. Add text, labels, and custom x-axis tick labels
ax.set_ylabel('Number of Samples')
ax.set_title('Dataset Sample Counts: Before vs. After Cleaning')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# 5. Attach a text label above each bar, displaying its count
ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

# 6. Save the figure
fig.tight_layout()
# plt.savefig('sample_counts_comparison.png', dpi=300)
# plt.show()
plt.close(fig)

print("Sample counts bar plot saved as 'sample_counts_comparison.png'")