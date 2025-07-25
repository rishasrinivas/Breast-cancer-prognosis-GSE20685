# Extract expression matrix and metadata
exprs_data = gse.pivot_samples("VALUE")
metadata = gse.phenotype_data

print("="*50)
print("DATASET OVERVIEW")
print("="*50)
print("Expression data shape:", exprs_data.shape)
print("Metadata shape:", metadata.shape)
print()

# Display metadata column names
print("METADATA COLUMNS:")
print("-"*20)
for i, col in enumerate(metadata.columns):
    print(f"{i+1:2d}. {col}")
print()

# Display first few rows of metadata to understand structure
print("FIRST 5 ROWS OF METADATA:")
print("-"*30)
print(metadata.head())
print()

# Display detailed info about each metadata column
print("DETAILED METADATA COLUMN INFO:")
print("-"*40)
for col in metadata.columns:
    unique_vals = metadata[col].unique()[:10]  # Show first 10 unique values
    print(f"\nColumn: {col}")
    print(f"Data type: {metadata[col].dtype}")
    print(f"Number of unique values: {metadata[col].nunique()}")
    print(f"Sample values: {unique_vals}")
print()

# Check for missing values in metadata
print("MISSING VALUES IN METADATA:")
print("-"*30)
missing_metadata = metadata.isnull().sum()
missing_metadata = missing_metadata[missing_metadata > 0]
print(missing_metadata)
print()

# Display expression data info
print("EXPRESSION DATA OVERVIEW:")
print("-"*25)
print("Expression data shape:", exprs_data.shape)
print("Sample gene names (first 10):")
print(exprs_data.index[:10].tolist())
print()
print("Sample patient IDs (first 10):")
print(exprs_data.columns[:10].tolist())
print()

# Check for missing values in expression data
print("MISSING VALUES IN EXPRESSION DATA:")
print("-"*35)
missing_expr_total = exprs_data.isnull().sum().sum()
print(f"Total missing values: {missing_expr_total}")
missing_per_sample = exprs_data.isnull().sum(axis=0)
print(f"Missing values per sample (first 10):")
print(missing_per_sample.head(10))
print()

# Check data types in expression data
print("EXPRESSION DATA TYPES:")
print("-"*22)
print(exprs_data.dtypes.value_counts())
print()

# Look for potential clinical outcome columns
print("POTENTIAL CLINICAL OUTCOME COLUMNS:")
print("-"*35)
potential_outcome_cols = []
for col in metadata.columns:
    if metadata[col].dtype == 'object':
        text_content = ' '.join(metadata[col].astype(str)).lower()
        if any(keyword in text_content for keyword in ['recur', 'surviv', 'death', 'event', 'status', 'outcome', 'relapse']):
            potential_outcome_cols.append(col)
            print(f"Column '{col}' might contain clinical outcomes")
            print(f"  Sample values: {metadata[col].unique()[:5]}")
            print()

# Check for numerical clinical variables
print("NUMERICAL CLINICAL VARIABLES:")
print("-"*30)
numerical_cols = metadata.select_dtypes(include=[np.number]).columns.tolist()
for col in numerical_cols:
    print(f"Column: {col}")
    print(f"  Range: {metadata[col].min()} to {metadata[col].max()}")
    print(f"  Missing: {metadata[col].isnull().sum()}")
    print()

# Check for time-to-event data
print("TIME-TO-EVENT DATA CHECK:")
print("-"*25)
time_related_cols = [col for col in metadata.columns if any(keyword in col.lower() for keyword in ['time', 'days', 'month', 'year', 'follow'])]
for col in time_related_cols:
    print(f"Column: {col}")
    print(f"  Data type: {metadata[col].dtype}")
    print(f"  Sample values: {metadata[col].unique()[:10]}")
    print()

# Display sample expression values
print("SAMPLE EXPRESSION VALUES:")
print("-"*25)
print("First 5 genes, first 5 samples:")
print(exprs_data.iloc[:5, :5])
print()

# Check for duplicate sample names
print("DUPLICATE SAMPLE CHECK:")
print("-"*22)
duplicates = metadata.index.duplicated().sum()
print(f"Duplicate sample IDs in metadata: {duplicates}")
duplicates_expr = exprs_data.columns.duplicated().sum()
print(f"Duplicate sample IDs in expression data: {duplicates_expr}")
print()

# Summary statistics for expression data
print("EXPRESSION DATA SUMMARY STATISTICS:")
print("-"*35)
expr_summary = exprs_data.describe()
print(expr_summary.iloc[:, :5])  # Show first 5 samples
