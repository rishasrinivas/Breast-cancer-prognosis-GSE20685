X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final,
    test_size=0.2,
    stratify=y_final,
    random_state=42
)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print("Training label distribution:")
print(y_train.value_counts())
print("Test label distribution:")
print(y_test.value_counts())
print()

# Step 6: Model Training
print("="*50)
print("STEP 6: MODEL TRAINING & EVALUATION")
print("="*50)

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Model Definitions
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

# Train and evaluate models
results = {}
model_objects = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    # Cross-validation
    scores = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=cv)
    results[name] = scores
    print(f"{name} - Mean AUC: {np.mean(scores):.3f} (+/- {np.std(scores)*2:.3f})")

    # Train final model
    model.fit(X_train, y_train)
    model_objects[name] = model

# Display results
results_df = pd.DataFrame(results)
print("\nCross-validation AUC Scores:")
print(results_df.describe())
print()
