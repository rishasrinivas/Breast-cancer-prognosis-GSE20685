# Top 1000 Most Variable Genes
print("Selecting top 1000 most variable genes...")
selector = SelectKBest(score_func=f_classif, k=1000)
X_selected = selector.fit_transform(X_scaled, y)
selected_indices_final = selector.get_support(indices=True)

genes_final = X_scaled_df.columns[selected_indices_final]

# Final dataset
X_final = pd.DataFrame(X_selected, columns=genes_final, index=X.index)
y_final = y.reset_index(drop=True)

print("Final Features Shape:", X_final.shape)
print()

# Check class balance
print("Class Balance in Final Dataset:")
print(y_final.value_counts())
print(f"Class 0 proportion: {sum(y_final==0)/len(y_final):.3f}")
print(f"Class 1 proportion: {sum(y_final==1)/len(y_final):.3f}")
print()
