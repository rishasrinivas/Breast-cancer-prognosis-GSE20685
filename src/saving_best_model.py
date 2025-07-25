import joblib

# Save the best model to a file
joblib.dump(best_model, 'best_model.joblib')

print("Best model (Logistic Regression) saved to best_model.joblib")
