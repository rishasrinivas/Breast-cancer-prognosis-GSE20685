# ROC Curve for Each Model
plt.figure(figsize=(10, 8))
for name, model in model_objects.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("roc_curves.png")
plt.show()

# Confusion Matrix for Best Model (based on mean CV AUC)
best_model_name = max(results, key=lambda x: np.mean(results[x]))
best_model = model_objects[best_model_name]

print(f"Best Model (based on CV AUC): {best_model_name}")

y_pred = best_model.predict(X_test)
y_proba_best = best_model.predict_proba(X_test)[:, 1]

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("confusion_matrix.png")
plt.show()

# Classification Report
print(f"\nClassification Report - {best_model_name}:")
print(classification_report(y_test, y_pred))

# Feature Importance (for Random Forest)
if best_model_name == "Random Forest":
    print("\nTop 20 Most Important Features:")
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Gene': X_final.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(20)

    print(feature_importance_df)

    # Feature Importance Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance_df, x='Importance', y='Gene', palette='viridis')
    plt.title("Top 20 Feature Importances - Random Forest")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()

# Model Performance Comparison
plt.figure(figsize=(10, 6))
model_names = list(results.keys())
auc_scores = [np.mean(results[model]) for model in model_names]
auc_stds = [np.std(results[model]) for model in model_names]

plt.bar(range(len(model_names)), auc_scores, yerr=auc_stds, capsize=5, alpha=0.7)
plt.xlabel("Models")
plt.ylabel("Mean AUC Score")
plt.title("Model Performance Comparison (Cross-Validation)")
plt.xticks(range(len(model_names)), model_names, rotation=45)
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("model_performance_comparison.png")
plt.show()

# Save final datasets
print("="*50)
print("FINAL DATASET SAVED")
print("="*50)
X_final.to_csv("processed_expression_data_GSE20685.csv", index=True)
y_final.to_csv("clinical_labels_death_GSE20685.csv", index=True)
print("Processed expression data saved to: processed_expression_data_GSE20685.csv")
print("Clinical labels saved to: clinical_labels_death_GSE20685.csv")

# Summary statistics
print("\nSUMMARY STATISTICS:")
print("-"*30)
print(f"Total samples: {X_final.shape[0]}")
print(f"Total features (genes): {X_final.shape[1]}")
print(f"Death events: {sum(y_final==1)} ({sum(y_final==1)/len(y_final)*100:.1f}%)")
print(f"Censored events: {sum(y_final==0)} ({sum(y_final==0)/len(y_final)*100:.1f}%)")
print(f"Best performing model: {best_model_name} (AUC: {np.mean(results[best_model_name]):.3f})")
