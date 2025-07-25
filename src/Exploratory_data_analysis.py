# Missing Values Check
print("Missing Values in Expression Data:")
missing_per_sample = X.isnull().sum(axis=1)
missing_per_gene = X.isnull().sum(axis=0)

print(f"Total missing values: {X.isnull().sum().sum()}")
print(f"Samples with missing values: {missing_per_sample[missing_per_sample > 0].shape[0]}")
print(f"Genes with missing values: {missing_per_gene[missing_per_gene > 0].shape[0]}")
print()

# Plot distribution of missing values per gene
plt.figure(figsize=(10, 4))
sns.histplot(missing_per_gene, bins=50)
plt.title("Distribution of Missing Values per Gene")
plt.xlabel("Missing Count")
plt.ylabel("Number of Genes")
plt.savefig("missing_values_per_gene.png")
plt.show()

# Variance Filtering
print("Variance Filtering:")
from sklearn.feature_selection import VarianceThreshold

# Check variance distribution
variances = X.var()
print(f"Mean variance: {variances.mean():.4f}")
print(f"Median variance: {variances.median():.4f}")
print(f"Min variance: {variances.min():.4f}")
print(f"Max variance: {variances.max():.4f}")

# Plot variance distribution
plt.figure(figsize=(10, 4))
sns.histplot(variances, bins=100)
plt.title("Distribution of Gene Expression Variances")
plt.xlabel("Variance")
plt.ylabel("Number of Genes")
plt.savefig("gene_expression_variances.png")
plt.show()

# Apply variance threshold
var_thresh = VarianceThreshold(threshold=0.01)
X_var_filtered = var_thresh.fit_transform(X)

print(f"After variance filtering (threshold=0.01): {X_var_filtered.shape}")
print()

# Normalization (Z-score)
print("Normalization:")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_var_filtered)

# Convert back to DataFrame for easier handling
selected_indices = var_thresh.get_support(indices=True)
genes_selected = X.columns[selected_indices]
X_scaled_df = pd.DataFrame(X_scaled, columns=genes_selected, index=X.index)

print("Normalized data shape:", X_scaled_df.shape)
print()
