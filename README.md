# Breast Cancer Prognosis Prediction using GSE20685 Gene Expression Data

This project aims to predict breast cancer patient prognosis (specifically, overall survival) using machine learning models trained on gene expression data from the publicly available GEO dataset **GSE20685**.

## Project Overview

*   **Dataset:** [GSE20685](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE20685) - Gene expression profiles and clinical metadata for 327 breast cancer patients.
*   **Goal:** Develop and evaluate machine learning models to predict patient death based on pre-treatment gene expression profiles.
*   **Key Finding:** A Logistic Regression model achieved excellent performance (AUC ~0.95) using the top 1000 most variable genes.

## Repository Structure
├── README.md # This file
├── requirements.txt # Python dependencies
├── src/
│ ├── data_download.py # (Optional if data is manually placed) Script to download GSE20685
│ ├── dataset_overview.py # Script to load data and print initial overview
│ ├── merging.py # Script to merge expression and clinical data
│ ├── exploratory_data_analysis.py # Script for EDA (missing values, variance)
│ ├── feature_selection.py # Script for variance filtering and selecting top K genes
│ ├── train_dataset.py # Script to split data and train models
│ ├── model_evaluation.py # Script to evaluate models and generate plots
│ └── saving_best_model.py # Script to save the best performing model and final data
├── results/
│ ├── processed_expression_data_GSE20685.csv # Final processed expression data
│ ├── clinical_labels_death_GSE20685.csv # Final clinical labels
│ └── best_model.pkl # Saved best model object (Logistic Regression)
└── reports/
  └── figures/ # Generated visualizations (PNG files)
    ├── missing_values_per_gene.png # EDA: Missing values distribution
    ├── gene_expression_variances.png # EDA: Gene variance distribution
    ├── model_performance_comparison.png # CV AUC comparison
    ├── roc_curves_comparison.png # ROC curves for test set
    ├── confusion_matrix_logreg.png # Confusion matrix for best model
    └── feature_importance_rf.png # Feature importance from Random Forest
  └── 6F.pdf #report

## Getting Started

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/breast-cancer-prognosis-gse20685.git
    cd breast-cancer-prognosis-gse20685
    ```

2.  **Install Dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    # Create a virtual environment (optional but recommended)
    python -m venv venv
    source venv/bin/activate # On Windows: venv\Scripts\activate

    # Install required packages
    pip install -r requirements.txt
    ```
   * Download the dataset from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE20685

3.  **Prepare the Data:**
    *   Download the **series matrix file (SOFT format)** for GSE20685 from GEO.
    *   Rename the downloaded file to `soft_file_GSE20685.txt`.
    *   Place this file in the `data/` directory.

4.  **Run the Analysis Pipeline:**
    Execute the Python scripts in the `src/` directory in the following order. You can run them one by one in your terminal or use a Jupyter Notebook (`main.ipynb`) to execute them sequentially.
    ```bash
    # Navigate to the src directory
    cd src

    # Run scripts in order
    python dataset_overview.py
    python merging.py
    python exploratory_data_analysis.py
    python feature_selection.py
    python train_dataset.py
    python model_evaluation.py
    python saving_best_model.py

    # Navigate back to the project root
    cd ..
    ```
    *Each script typically loads data from `data/` or `results/`, processes it, and saves its output to `results/` or generates plots in `reports/figures/`.*

5.  **View Results:**
    *   Processed data (CSV files) will be saved in `results/`.
    *   The best trained model (`best_model.pkl`) will be in `results/`.
    *   Generated figures are in `reports/figures/`.
    *   (Optional) A summary report can be placed in the main directory or `reports/`.

## Data Source

*   **GSE20685:** Wang, S. C., et al. (2011). Gene expression profiling of breast cancer cells with different tumorigenicity and metastatic potential. *Journal of Biomedical Science*, 18(1), 1-13. Data accessed from NCBI GEO.

## Tools & Libraries

*   **Data Acquisition/Processing:** `GEOparse`, `pandas`, `numpy`
*   **Machine Learning:** `scikit-learn`
*   **Visualization:** `matplotlib`, `seaborn`
*   **Environment:** `Python 3.x`

## Authors

*  Risha Reddy Mukkisa- Initial work

