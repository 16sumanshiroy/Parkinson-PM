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
![1000000894](https://github.com/user-attachments/assets/7c025757-a738-4c41-9c72-50972617ada8)
![1000000895](https://github.com/user-attachments/assets/43b49c49-8669-48af-af0a-d4e45cc4177a)




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
![1000000894](https://github.com/user-attachments/assets/7c025757-a738-4c41-9c72-50972617ada8)
![1000000895](https://github.com/user-attachments/assets/43b49c49-8669-48af-af0a-d4e45cc4177a)

### 🔥 Correlation Heatmap
Shows relationships between features and helps in dimensionality reduction.
![1000000892](https://github.com/user-attachments/assets/b71c37ed-d077-4298-9b49-0ba565e50f35)

### ✅ Confusion Matrix
Visualizes True Positives, False Positives, etc.
![1000000893](https://github.com/user-attachments/assets/2f1c648e-490b-4f79-86c9-ba11329ba4de)


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
- **Sumanshi Roy**
- Ramandeep Ratan  
- Anisha Yadav  
- Musharraf Ali  
> Supervised by **Mr. Santosh Kumar**, Asst. Professor, REC Bijnor

---

## 📚 Citation
Presented at **ICAISI-2025** | Scopus-Indexed | Accepted for publication in **Taylor & Francis / CRC Press**
![1000000905](https://github.com/user-attachments/assets/b5827d00-4d20-42a2-96e7-2e2aa6ed90c9)



---

## 📎 License
This academic project is open for educational reuse with attribution.

