print("STEP 2: MERGING EXPRESSION AND CLINICAL LABELS")
print("="*50)

# Use event_death as the primary outcome
# Convert to numeric and handle missing values
metadata['death_event'] = pd.to_numeric(metadata['characteristics_ch1.3.event_death'], errors='coerce')

# Check the distribution of death events
print("Death Event Distribution:")
print(metadata['death_event'].value_counts(dropna=False))
print()

# Drop rows with missing death event labels
metadata_clean = metadata.dropna(subset=['death_event'])

# Match samples between expression and metadata
common_samples = list(set(exprs_data.columns) & set(metadata_clean.index))
exprs_filtered = exprs_data[common_samples]
labels = metadata_clean.loc[common_samples, 'death_event'].astype(int)

# Transpose for easier handling (samples as rows, genes as columns)
X = exprs_filtered.T
y = labels

print("Final dataset shape (samples x features):", X.shape)
print("Label distribution:")
print(y.value_counts())
print()
