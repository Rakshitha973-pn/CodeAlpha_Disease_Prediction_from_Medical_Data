
# Disease Prediction from Medical Data

## 📋 Project Overview
This project implements machine learning classification models to predict the possibility of diseases based on patient medical data. Built as part of **Code Alpha ML Internship - Task 4**.

## 🎯 Objective
Predict disease occurrence (specifically breast cancer) using structured medical datasets and compare the performance of multiple classification algorithms.

## 📊 Dataset
**Breast Cancer Wisconsin (Diagnostic) Dataset**
- **Source**: UCI Machine Learning Repository / Scikit-learn
- **Samples**: 569 patients
- **Features**: 30 numeric features
- **Target Variable**: 
  - 0 = Malignant (Cancerous)
  - 1 = Benign (Non-cancerous)

### Features Include:
- Radius, texture, perimeter, area
- Smoothness, compactness, concavity
- Concave points, symmetry, fractal dimension
- Mean, standard error, and worst values for each measurement

## 🛠️ Technologies Used
- **Python 3.x**
- **Libraries**:
  - `numpy` - Numerical computations
  - `pandas` - Data manipulation
  - `matplotlib` & `seaborn` - Data visualization
  - `scikit-learn` - Machine learning algorithms
  - `xgboost` - Gradient boosting classifier

## 🤖 Machine Learning Models Implemented
1. **Logistic Regression** - Baseline linear model
2. **Support Vector Machine (SVM)** - Kernel-based classifier
3. **Random Forest** - Ensemble learning method
4. **XGBoost** - Advanced gradient boosting

## 📁 Project Structure
```
CodeAlpha_Disease_Prediction_from_Medical_Data /
│
├── Disease_Prediction_from_Medical_Data.py   # Main Python script
├── breast_cancer_data.csv                     # Dataset file
├── README.md                                  # Project documentation
│
└── outputs/
    ├── model_results.csv                      # Performance metrics
    ├── correlation_heatmap.png                # Feature correlations
    ├── target_distribution.png                # Class distribution
    ├── model_comparison.png                   # Model performance comparison
    ├── roc_curves.png                         # ROC-AUC curves
    ├── confusion_matrix_best.png              # Confusion matrix
    └── feature_importance.png                 # Top features visualization
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- Anaconda (recommended) or pip

### Install Required Libraries
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

Or using conda:
```bash
conda install numpy pandas matplotlib seaborn scikit-learn
conda install -c conda-forge xgboost
```

## ▶️ How to Run

1. **Clone the repository**
```bash
git clone https://github.com/your-username/disease-prediction.git
cd disease-prediction
```

2. **Run the Python script**
```bash
python Disease_Prediction_from_Medical_Data.py
```

Or open in **Jupyter Notebook**:
```bash
jupyter notebook Disease_Prediction_from_Medical_Data.ipynb
```

3. **View Results**
- Console output shows model performance metrics
- Generated PNG files contain visualizations
- `model_results.csv` contains a detailed comparison

## 📈 Key Features

### 1. Exploratory Data Analysis (EDA)
- Statistical summary of features
- Correlation heatmap analysis
- Target distribution visualization
- Missing value detection

### 2. Data Preprocessing
- Train-test split (80-20 ratio)
- Feature scaling using StandardScaler
- Stratified sampling for balanced classes

### 3. Model Training & Evaluation
- Cross-validation support
- Multiple evaluation metrics:
  - **Accuracy**: Overall correctness
  - **Precision**: Positive prediction accuracy
  - **Recall**: True positive detection rate
  - **F1-Score**: Harmonic mean of precision & recall
  - **ROC-AUC**: Area under ROC curve

### 4. Visualizations
- Model performance comparison charts
- ROC curves for all models
- Confusion matrix for best model
- Feature importance analysis
- Correlation heatmaps

### 5. Feature Importance
- Identifies most influential features
- Random Forest-based importance ranking
- Visual representation of top features

## 📊 Results

Sample output (results may vary):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.9825 | 0.9859 | 0.9859 | 0.9859 | 0.9977 |
| Support Vector Machine | 0.9825 | 0.9859 | 0.9859 | 0.9859 | 0.9977 |
| Random Forest | 0.9649 | 0.9718 | 0.9718 | 0.9718 | 0.9954 |
| XGBoost | 0.9737 | 0.9859 | 0.9718 | 0.9788 | 0.9954 |

🏆 **Best Model**: Logistic Regression / SVM (98.25% accuracy)

## 🔍 Model Insights

### Top 5 Most Important Features:
1. worst perimeter
2. worst concave points
3. mean concave points
4. worst radius
5. mean perimeter

### Key Findings:
- All models achieved >96% accuracy
- SVM and Logistic Regression performed best
- Feature scaling significantly improved performance
- Worst/mean measurements are most predictive

## 📝 Code Highlights

```python
# Model training example
models = {
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'SVM': SVC(probability=True),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'XGBoost': XGBClassifier()
}

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
```

## 🎓 Learning Outcomes
- Implemented multiple ML classification algorithms
- Performed comprehensive EDA and preprocessing
- Evaluated models using various metrics
- Created professional visualizations
- Analyzed feature importance
- Compared model performance effectively

## 📚 References
- [UCI ML Repository - Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

## 🤝 Contributing
This is an internship project for Code Alpha. Suggestions and improvements are welcome!

## 👨‍💻 Author
Rakshitha PN  
Code Alpha ML Intern  
Task 4: Disease Prediction from Medical Data

## 📧 Contact
- Email: rakshithapn123@gmail.com
- LinkedIn: https://www.linkedin.com/in/rakshitha-pn-b305a2292
- GitHub: https://github.com/Rakshitha973-pn

## 📄 License
This project is created for educational purposes as part of Code Alpha ML Internship.

## 🙏 Acknowledgments
- Code Alpha for the internship opportunity
- UCI Machine Learning Repository for the dataset
- Scikit-learn and XGBoost communities

