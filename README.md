# 🧠 An Efficient Ensemble-Based Model for Prediction of Neurodegenerative Malady and Acute Encephalopathy

A machine learning project focused on the early detection of **Parkinson’s** and related neurodegenerative disorders using ensemble techniques and vocal biomarkers. This project aims to assist in timely diagnosis through non-invasive, interpretable, and accurate predictive models.

---

## 📌 Problem Statement
Neurodegenerative diseases like Parkinson’s and Alzheimer’s are hard to diagnose early due to overlapping symptoms and data complexity. Our project improves diagnostic accuracy using ensemble learning models that analyze vocal and biomedical features.

---

## 🧪 Dataset
- **Source:** UCI Machine Learning Repository – [Parkinson’s Telemonitoring Dataset](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring)
- **Features:** Jitter, Shimmer, HNR, RPDE, PPE, etc.
- **Target Variable:** `status` (1 = Parkinson’s, 0 = Healthy)

---

## 🛠️ Technologies & Libraries
- **Languages:** Python
- **Libraries:** `NumPy`, `Pandas`, `Scikit-learn`, `Matplotlib`, `Seaborn`, `SHAP`, `SMOTE`
- **Platforms:** Jupyter Notebook, Google Colab

---

## 🧠 Models Implemented

### 📈 Machine Learning:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Kernel SVM
- Decision Tree
- Naive Bayes

### 🔀 Ensemble Learning:
- Random Forest
- XGBoost ✅ *(Best performance: 98% accuracy)*
- AdaBoost
- Gradient Boosting
- Stacking

---

## ⚙️ Workflow & Methodology
1. **Preprocessing:** Missing values, scaling, outlier removal
2. **Data Balancing:** SMOTE applied
3. **Feature Engineering:** PCA, SHAP, Correlation filtering
4. **Model Training:** Cross-validation + hyperparameter tuning
5. **Evaluation:** Accuracy, F1-score, ROC-AUC, Confusion Matrix
6. **Visualization:** Data insights and model performance

---

## 📸 Sample Outputs & Plots

### 🎯 Accuracy Comparison of Models
![Accuracy Comparison](images/accuracy.png)

### 🔥 Correlation Heatmap
Shows relationships between features and helps in dimensionality reduction.
![Correlation Heatmap](images/heatmap.png)

### ✅ Confusion Matrix
Visualizes True Positives, False Positives, etc.
![Confusion Matrix](images/confusion_matrix.png)

---

## 📊 Results Summary
| Model              | Accuracy |
|--------------------|----------|
| XGBoost            | **98%**  |
| Random Forest      | 95%      |
| Stacking Classifier| 92%      |
| Decision Tree      | 89%      |
| KNN                | 82%      |

---

## 🚀 Future Scope
- Integrate with real-time hospital data
- Deploy as a web/mobile diagnostic tool
- Enhance interpretability using SHAP/LIME
- Extend dataset to include imaging/genetics
- Personalize predictions for treatment plans

---

## 👩‍💻 Authors
- Anisha Yadav  
- Ramandeep Ratan  
- **Sumanshi Roy** — [GitHub](https://github.com/sumanshiroy)  
- Musharraf Ali  
> Supervised by **Mr. Santosh Kumar**, Asst. Professor, REC Bijnor

---

## 📚 Citation
Presented at **ICAISI-2025** | Scopus-Indexed | Accepted for publication in **Taylor & Francis / CRC Press**

---

## 📎 License
This academic project is open for educational reuse with attribution.

