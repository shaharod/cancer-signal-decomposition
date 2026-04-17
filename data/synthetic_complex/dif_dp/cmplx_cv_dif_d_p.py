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

threshold = 0.65
max_diff = 2

# loading real data
df_real_healthy = pd.read_csv(healthy_path, index_col=0)
df_real_cancerA = pd.read_csv(diseaseA_path, index_col=0)
df_real_cancerB = pd.read_csv(diseaseB_path, index_col=0)

df_clean_cancerA = clean_rows(df_real_cancerA.T).T
df_clean_cancerB = clean_rows(df_real_cancerB.T).T

print(f"Cancer A samples before: {df_real_cancerA.shape[1]}, after cleaning: {df_clean_cancerA.shape[1]}")
print(f"Cancer B samples before: {df_real_cancerB.shape[1]}, after cleaning: {df_clean_cancerB.shape[1]}")
num = 1000
rows_over_num_a= df_real_cancerA.index[(df_real_cancerA > num).any(axis=1)]

print(f"Rows of cancer a with values > {num}:", rows_over_num_a.tolist())
rows_over_num_b= df_real_cancerB.index[(df_real_cancerB > num).any(axis=1)]

print(f"Rows of cancer b with values > {num}:", rows_over_num_b.tolist())

rows_over_num_a= df_clean_cancerA.index[(df_clean_cancerA > num).any(axis=1)]

print(f"Rows of clean cancer a with values > {num}:", rows_over_num_a.tolist())
rows_over_num_b= df_clean_cancerB.index[(df_clean_cancerB > num).any(axis=1)]

print(f"Rows of clean cancer b with values > {num}:", rows_over_num_b.tolist())

# loading thetas
metadata_A = pd.read_csv(theta_A_path)
metadata_B = pd.read_csv(theta_B_path)

dropped_samples_A = set(df_real_cancerA.columns) - set(df_clean_cancerA.columns)
dropped_samples_B = set(df_real_cancerB.columns) - set(df_clean_cancerB.columns)

# # Find high-theta samples in the RAW metadata
# raw_high_theta_A = set(metadata_A[metadata_A['data_list'] >= threshold]['Unnamed: 0'])
# raw_high_theta_B = set(metadata_B[metadata_B['data_list'] >= threshold]['Unnamed: 0'])

# # # Intersect to see how many high-theta samples we lost to the cleaning function
# # lost_high_theta_A = raw_high_theta_A.intersection(dropped_samples_A)
# # lost_high_theta_B = raw_high_theta_B.intersection(dropped_samples_B)

# # print(f"\n--- QC: Missed Opportunities ---")
# # print(f"Disease A: Lost {len(lost_high_theta_A)} high-theta samples during cleaning.")
# # print(f"Disease B: Lost {len(lost_high_theta_B)} high-theta samples during cleaning.")
# # # Note: If this number is high, you may need to adjust `clean_rows` to prefer keeping 
# # # the duplicate with the highest theta, rather than random/first dropping.
# # # ---------------------------------------------------------
# # # Extract and Print the Lost High-Theta Samples
# # # ---------------------------------------------------------

# # # # Filter the raw metadata to only include the lost samples
# # # lost_details_A = metadata_A[metadata_A['Unnamed: 0'].isin(lost_high_theta_A)]
# # # lost_details_B = metadata_B[metadata_B['Unnamed: 0'].isin(lost_high_theta_B)]

# # # # Rename columns just for a cleaner printout
# # # lost_details_A = lost_details_A.rename(columns={'Unnamed: 0': 'Sample_ID', 'data_list': 'Theta'})
# # # lost_details_B = lost_details_B.rename(columns={'Unnamed: 0': 'Sample_ID', 'data_list': 'Theta'})

# # # print("\n--- Details of Lost High-Theta Samples: Disease A ---")
# # # if lost_details_A.empty:
# # #     print("No high-theta samples lost! Perfect.")
# # # else:
# # #     # Sort by Theta descending so you see the biggest losses first
# # #     lost_details_A = lost_details_A.sort_values(by='Theta', ascending=False)
# # #     print(lost_details_A[['Sample_ID', 'Theta']].to_string(index=False))

# # # print("\n--- Details of Lost High-Theta Samples: Disease B ---")
# # # if lost_details_B.empty:
# # #     print("No high-theta samples lost! Perfect.")
# # # else:
# # #     lost_details_B = lost_details_B.sort_values(by='Theta', ascending=False)
# # #     print(lost_details_B[['Sample_ID', 'Theta']].to_string(index=False))

valid_samples_A = df_clean_cancerA.columns
valid_samples_B = df_clean_cancerB.columns

# Filter the metadata DataFrames using .isin()
# This ensures we only keep the theta values for samples that still exist in our clean matrices
metadata_A = metadata_A[metadata_A['Unnamed: 0'].isin(valid_samples_A)].copy()
metadata_B = metadata_B[metadata_B['Unnamed: 0'].isin(valid_samples_B)].copy()

# Reset the index of the metadata so it's clean (optional but good practice)
metadata_A.reset_index(drop=True, inplace=True)
metadata_B.reset_index(drop=True, inplace=True)


# Rename the column back to something sensible for clarity
metadata_A.rename(columns={'index': 'Sample_ID'}, inplace=True)
metadata_B.rename(columns={'index': 'Sample_ID'}, inplace=True)
print(f"Metadata A aligned. Remaining thetas: {len(metadata_A)}")
print(f"Metadata B aligned. Remaining thetas: {len(metadata_B)}")
# assert list(metadata_A['Sample_ID']) == list(df_clean_cancerA.columns), "FATAL: Disease A Metadata and Matrix are misaligned!"
# assert list(metadata_B['Sample_ID']) == list(df_clean_cancerB.columns), "FATAL: Disease B Metadata and Matrix are misaligned!"

# Creating Profiles
# average healthy data
blueprint_healthy = df_real_healthy.mean(axis=1).values
print(blueprint_healthy)
print(f"blueprint healthy sum is: {blueprint_healthy.sum()}")


# --- Disease A (CRC) ---
# Filter metadata for high theta samples
high_theta_A_meta = metadata_A[metadata_A['data_list'] >= threshold]

# print(f"Found {len(samples_A)} Disease A samples with theta >= {threshold}")




# --- Disease B (SCLC) ---
# Filter metadata for high theta samples
high_theta_B_meta = metadata_B[metadata_B['data_list'] >= threshold]

# print(f"Found {len(samples_B)} Disease B samples with theta >= {threshold}")


n_A = len(high_theta_A_meta)
n_B = len(high_theta_B_meta)

print(f"Initial high-theta samples found -> Disease A: {n_A}, Disease B: {n_B}")

# 2. Apply Diversity Balancing Constraint
min_n = min(n_A, n_B)
allowed_max = min_n + max_diff

if n_A > allowed_max:
    # Downsample A randomly to preserve diversity, or use .nlargest() to prioritize purity
    high_theta_A_meta = high_theta_A_meta.sample(n=allowed_max, random_state=42)
    print(f"Downsampled Disease A from {n_A} to {allowed_max} to balance diversity.")
elif n_B > allowed_max:
    # Downsample B randomly 
    high_theta_B_meta = high_theta_B_meta.sample(n=allowed_max, random_state=42)
    print(f"Downsampled Disease B from {n_B} to {allowed_max} to balance diversity.")

samples_A = high_theta_A_meta['Unnamed: 0'].values
# Extract their mixed profiles
mixed_A_subset = df_clean_cancerA[samples_A].values
thetas_A_real = high_theta_A_meta['data_list'].values
# Deconvolute EACH sample to get its pure disease profile
# We use broadcasting [:, None] to align the 1D arrays with the 2D matrix
pure_disease_A_matrix = (mixed_A_subset - (1 - thetas_A_real) * blueprint_healthy[:, None]) / thetas_A_real
pure_disease_A_matrix = pure_disease_A_matrix.clip(min=0)

# 1. Create a boolean mask of the rows that meet your criteria in the NumPy array
mask = (pure_disease_A_matrix > num).any(axis=1)

# 2. Apply that mask to the index of your original Pandas DataFrame
actual_genes_a = df_clean_cancerA[samples_A].index[mask]

# 3. Print the resulting list of gene names
print(f"Genes over {num} in pure A after dividing theta:")
print(actual_genes_a.tolist())

samples_B = high_theta_B_meta['Unnamed: 0'].values
thetas_B_real = high_theta_B_meta['data_list'].values
# Deconvolute EACH sample to get its pure disease profile
# Extract their mixed profiles
mixed_B_subset = df_clean_cancerB[samples_B].values
pure_disease_B_matrix = (mixed_B_subset - (1 - thetas_B_real) * blueprint_healthy[:, None]) / thetas_B_real
pure_disease_B_matrix = pure_disease_B_matrix.clip(min=0)

# 1. Create a boolean mask of the rows that meet your criteria in the NumPy array
mask = (pure_disease_B_matrix > num).any(axis=1)

# 2. Apply that mask to the index of your original Pandas DataFrame
actual_genes_b = df_clean_cancerB[samples_B].index[mask]

# 3. Print the resulting list of gene names
print(f"Genes over {num} in pure B after dividing theta:")
print(actual_genes_b.tolist())
print("Successfully isolated pure profiles for multiple high-theta Disease A and B samples.")


### Pure datasets
n_genes = len(blueprint_healthy)

n_healthy_samples = 300
n_disease_A_samples = 200
n_disease_B_samples = 200
bio_cv = 0.1
healthy_std = bio_cv * blueprint_healthy
total_healthy_needed = n_healthy_samples + n_disease_A_samples + n_disease_B_samples

healthy_pool = np.random.normal(
    blueprint_healthy[:, None], healthy_std[:, None], size=(n_genes, total_healthy_needed)
).clip(min=0)

# Randomly sample columns (patient profiles) from our pure matrices to reach the required count.
# We use replace=True in case we need more samples than we filtered out.
idx_A = np.random.choice(pure_disease_A_matrix.shape[1], size=n_disease_A_samples, replace=True)
base_disease_A = pure_disease_A_matrix[:, idx_A]

idx_B = np.random.choice(pure_disease_B_matrix.shape[1], size=n_disease_B_samples, replace=True)
base_disease_B = pure_disease_B_matrix[:, idx_B]

# Apply biological jitter to simulate slight variations (even if a patient is sampled twice)
diseaseA_std = bio_cv * base_disease_A
disease_A_pool = np.random.normal(base_disease_A, diseaseA_std, size=(n_genes, n_disease_A_samples)).clip(min=0)

diseaseB_std = bio_cv * base_disease_B
disease_B_pool = np.random.normal(base_disease_B, diseaseB_std, size=(n_genes, n_disease_B_samples)).clip(min=0)

print(f"Disease pools created: Disease A ({disease_A_pool.shape}), Disease B ({disease_B_pool.shape})")

pure_healthy_data = healthy_pool[:, :n_healthy_samples]

print(f"Pools created: Healthy ({healthy_pool.shape}), Disease A ({disease_A_pool.shape}), Disease B ({disease_B_pool.shape})")
def apply_sequencing_noise(clean_matrix):
    """
    Func to apply poisson technical noise. Does so by anchoring the variance to sequencing depth (0.5M to 5M).
    Returns the normalized, noisy matrix.
    """
    n_samples = clean_matrix.shape[1]
    target_depths = np.random.uniform(500_000, 5_000_000, size=n_samples)
    scaling_factors = (target_depths / 1_000_000).reshape(1, -1) ## shape 1 row n cols
    ## example: if we get 3M for target depth, the scaling is 3 (since sum is roughly 1M) 
    
    expected_counts = clean_matrix * scaling_factors ## multiply by the scaling
    raw_synthetic_counts = np.random.poisson(lam=expected_counts) ## apply our poisson
    
    # Normalize back to stabilize the variance
    return raw_synthetic_counts / scaling_factors ## normalize by the scaling factor

# # Mixing
n_mix = n_disease_A_samples ##NOTE for now its fine, num of A and B samples is the same

# mix disease A
thetas_A = np.random.uniform(0,1,n_mix)
healthy_data_samples_A = healthy_pool[:, n_healthy_samples:n_healthy_samples+n_mix] # healthy bio noisy parts for A samples

mixed_A_clean = (1 - thetas_A) * healthy_data_samples_A + thetas_A * disease_A_pool # clean := no sample/tech noise

# mix disease B
thetas_B = np.random.uniform(0, 1, n_mix)
healthy_data_samples_B = healthy_pool[:, n_mix + n_healthy_samples:]

mixed_B_clean = (1 - thetas_B) * healthy_data_samples_B + thetas_B * disease_B_pool

print(f"Created mixtures. Mix A shape: {mixed_A_clean.shape}, Mix B shape: {mixed_B_clean.shape}")
print(f"healthy data samples: {healthy_data_samples_A.shape}")

## applying tech noise
final_mixed_A = apply_sequencing_noise(mixed_A_clean)
final_mixed_B = apply_sequencing_noise(mixed_B_clean)
final_healthy = apply_sequencing_noise(pure_healthy_data)
## Saving

df_healthy = pd.DataFrame(final_healthy, index=df_real_healthy.index, 
                          columns=[f'Healthy-Sample{i}' for i in range(pure_healthy_data.shape[1])])
df_healthy.to_csv('healthy_data.csv')

# 3. Save Pure Disease Profiles
df_pure_A = pd.DataFrame(disease_A_pool, index=df_real_healthy.index, 
                         columns=[f'DiseaseA-Sample{i}' for i in range(disease_A_pool.shape[1])])

rows_over_num_a_pure_cv= df_pure_A.index[(df_pure_A > num).any(axis=1)]
print(f"Genes over {num} for pure disease A after cv noise")
print(rows_over_num_a_pure_cv.to_list())

# df_pure_A.to_csv('pure_disease_A.csv')

df_pure_B = pd.DataFrame(disease_B_pool, index=df_real_healthy.index, 
                         columns=[f'DiseaseB-Sample{i}' for i in range(disease_B_pool.shape[1])])
# df_pure_B.to_csv('pure_disease_B.csv')
rows_over_num_b_pure_cv= df_pure_B.index[(df_pure_B > num).any(axis=1)]
print(f"Genes over {num} for pure disease B after cv noise")
print(rows_over_num_b_pure_cv.to_list())

# 4. Save Mixed Data A and its Theta values
df_mixed_A = pd.DataFrame(final_mixed_A, index=df_real_healthy.index, 
                          columns=[f'DiseaseA-Sample{i}' for i in range(final_mixed_A.shape[1])])
# df_mixed_A.to_csv('mixed_data_A.csv')
# pd.Series(thetas_A, name='Theta').to_csv('mixed_A_thetas.csv', index=False)
rows_over_num_a_mix_noise= df_mixed_A.index[(df_mixed_A > num).any(axis=1)]
print(f"Genes over {num} for mix disease A after all noise")
print(rows_over_num_a_mix_noise.to_list())
# 5. Save Mixed Data B and its Theta values
df_mixed_B = pd.DataFrame(final_mixed_B, index=df_real_healthy.index, 
                          columns=[f'DiseaseB-Sample{i}' for i in range(final_mixed_B.shape[1])])
rows_over_num_b_mix_noise= df_mixed_B.index[(df_mixed_B > num).any(axis=1)]
print(f"Genes over {num} for mix disease B after all noise")
print(rows_over_num_b_mix_noise.to_list())
raise
# df_mixed_B.to_csv('mixed_data_B.csv')
# pd.Series(thetas_B, name='Theta').to_csv('mixed_B_thetas.csv', index=False)

# print(f"All components saved")
# SAVE COMBINED DATASETS

labels_A = np.ones(len(thetas_A), dtype=int)
labels_B = np.full(len(thetas_B), 2, dtype=int)
all_labels = np.concatenate([labels_A, labels_B])

## save one combined csv of the pure disease truth of each disease sample (a and b)
combined_pure_df = pd.concat([df_pure_A, df_pure_B], axis=1)
combined_pure_df.to_csv("pure_disease_truth.csv")
print(combined_pure_df.shape)

## save theta values of all disease samples in one file
all_thetas = np.concatenate([thetas_A, thetas_B])

# Extract the exact sample names you generated earlier to ensure perfect alignment
sample_names = list(df_mixed_A.columns) + list(df_mixed_B.columns)

# Build the combined metadata DataFrame
combined_theta_df = pd.DataFrame({
    'theta_value': all_thetas,
    'disease_type': all_labels
}, index=sample_names)

combined_theta_df.to_csv("theta_values.csv", index=True)
print(f"Combined Theta Shape: {combined_theta_df.shape}")
# =========================================================
# 1. COMBINE AND SAVE: Uniform Theta Mixed Data
# =========================================================
# We use axis=1 to stack the 150 'B' patients next to the 150 'A' patients
combined_mixed_uniform_df = pd.concat([df_mixed_A, df_mixed_B], axis=1)
combined_mixed_uniform_df.loc['disease_type'] = all_labels
combined_mixed_uniform_df.to_csv("disease_data_uniform_theta.csv")

print(f"Saved Uniform Theta Mix. Shape: {combined_mixed_uniform_df.shape}") 
# Expected: (20000, 300)

# =========================================================
# 2. GENERATE, COMBINE, AND SAVE: Fixed Theta (0.5) Mixed Data
# =========================================================
fixed_theta = 0.5

# Mix the data using the fixed 0.5 ratio
# We reuse healthy_data_samples_A/B so the background noise is consistent!
mixed_A_05_clean = (1 - fixed_theta) * healthy_data_samples_A + fixed_theta * disease_A_pool
mixed_B_05_clean = (1 - fixed_theta) * healthy_data_samples_B + fixed_theta * disease_B_pool

final_mixed_A_05 = apply_sequencing_noise(mixed_A_05_clean)
final_mixed_B_05 = apply_sequencing_noise(mixed_B_05_clean)
# Convert these new matrices into Pandas DataFrames
df_mixed_A_05 = pd.DataFrame(
    final_mixed_A_05, 
    index=df_real_healthy.index, 
    columns=[f'DiseaseA-Sample{i}' for i in range(final_mixed_A_05.shape[1])]
)

df_mixed_B_05 = pd.DataFrame(
    final_mixed_B_05, 
    index=df_real_healthy.index, 
    columns=[f'DiseaseB-Sample{i}' for i in range(final_mixed_B_05.shape[1])]
)

# Combine them side-by-side
combined_mixed_05_df = pd.concat([df_mixed_A_05, df_mixed_B_05], axis=1)
combined_mixed_05_df.loc['disease_type'] = all_labels
combined_mixed_05_df.to_csv("disease_data_theta05.csv")

print(f"Saved Fixed Theta (0.5) Mix. Shape: {combined_mixed_05_df.shape}")
# Expected: (20000, 400)

## Raw Data PCA Baseline (or Pre-Training Sanity Check)
from sklearn.decomposition import PCA

# 1. Prep the Uniform Theta data (drop the label row and transpose so samples are rows)
raw_uniform_data = combined_mixed_uniform_df.drop('disease_type').values.T 

# 2. Run PCA directly on the raw, noisy sequence counts
pca_coords_uniform = PCA(n_components=2).fit_transform(raw_uniform_data)

# 3. Plot it using the custom functions at the top of your script
# Coloring by 'all_thetas' to see the continuous gradients (the "lines")
ax, scat = scatter(pca_coords_uniform[:, 0], pca_coords_uniform[:, 1], 
                   c=all_thetas, cmap='magma', return_scat_obj=True, diag=False)

# Add a colorbar and labels
plt.colorbar(scat, label="Theta (Tumor Fraction)")
show(title="Raw Data PCA: Uniform Theta (The 'Lines')", 
     xlabel="PC1", ylabel="PC2", aspect=True)
## Raw Data PCA: Uniform Theta by Disease Type
# 1. Plot using disease_type (all_labels) instead of theta
# Set diag=False to remove the weird black line!
ax, scat = scatter(pca_coords_uniform[:, 0], pca_coords_uniform[:, 1], 
                   c=all_labels, cmap='tab10', diag=False, return_scat_obj=True)

# 2. Add a legend for the discrete disease types
# Since tab10 is categorical, a legend works better than a colorbar
handles, _ = scat.legend_elements(prop="colors")
ax.legend(handles, ["Disease A (CRC)", "Disease B (SCLC)"], title="Disease Type")

show(title="Raw Data PCA: Uniform Theta by Disease Type", 
     xlabel="PC1", ylabel="PC2", aspect=True)
