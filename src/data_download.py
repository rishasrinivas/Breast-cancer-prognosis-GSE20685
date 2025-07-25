# Install required packages if not already installed
!pip install GEOparse scikit-learn seaborn matplotlib pandas numpy

import GEOparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Download dataset
gse = GEOparse.get_GEO(geo="GSE20685", destdir="./data")

# Extract expression matrix and metadata
exprs_data = gse.pivot_samples("VALUE")
metadata = gse.phenotype_data

print("Expression shape:", exprs_data.shape)
print("Metadata shape:", metadata.shape)
