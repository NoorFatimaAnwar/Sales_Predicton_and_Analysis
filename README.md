# 📊 Sales Price Prediction & Analysis

This project predicts **product sales based on advertising budgets** (TV, Radio, Newspaper) using machine learning models.  
It also provides **feature importance analysis**, **what-if budget scenarios**, and an **interactive prediction tool**.

---

## 🚀 Features

- **Baseline Model** – Simple mean-based predictor  
- **ML Models** – Linear Regression, Ridge Regression, Support Vector Machine (SVM)  
- **Evaluation** – Metrics (MAE, RMSE, R²) + Actual vs Predicted plots  
- **Feature Importance** – Using permutation importance  
- **Scenario Analysis** – Simulate budget changes (e.g., +10% TV, -20% Newspaper)  
- **Interactive Prediction** – Enter your own ad spend to predict sales  

---

## 🗂️ Project Structure
├── data/

│ └── Advertising.csv # Dataset

├── src/

│ ├── preprocess.py # Load, split, scale data
│ ├── baseline.py # Baseline model
│ ├── models.py # Train models (LR, Ridge, SVM)
│ ├── evaluation.py # Metrics + plots
│ ├── feature_importance.py # Permutation importance
│ ├── predict.py # Predict sales for user inputs
│ ├── scenario.py # Budget adjustment scenarios

├── notebook/

  ├── Sales_Prediction.ipynb # Main notebook (analysis + demo)
  
└── README.md # Project documentation

---

## 📦 Dependencies

- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- ipywidgets (for interactive input in Jupyter Notebook)  

---

## ▶️ Usage

### 1. Run Jupyter Notebook
```bash
jupyter notebook Sales_Prediction.ipynb
```
### 2. Predict Sales (Interactive)
Inside the notebook, enter **TV, Radio, and Newspaper ad spends** → get predicted sales instantly.

### 3. Scenario Analysis
Test budget changes like:
- 📺 **+10% TV**  
- 📰 **-20% Newspaper**  
- 📻 **+10% Radio**  
- 📺 **+10% TV** & 📻 **+15% Radio**  
- 💰 **-10% All spends**  

---

## 📈 Results

- **Baseline model** → acts as benchmark  
- **SVM model** → best performance (lowest RMSE, highest R²)  
- **TV & Radio** → most important features  
- **TV + Radio budgets** → biggest improvement in predicted sales  

---

## 📊 Business Insights & Recommendations

Based on scenario testing with the trained model:

- 📺 **TV advertising has the strongest impact on sales**  
  +10% TV spend → **+0.53 higher average sales**  

- 📻 **Radio advertising is highly effective**  
  +10% Radio spend → **+0.44 higher sales**  

- 🔗 **Combining TV and Radio yields the best results**  
  +10% TV & +15% Radio → **+1.22 higher sales** (synergy effect)  

- 📰 **Newspaper advertising has minimal effect**  
  –20% Newspaper spend → only **–0.02 change in sales**  

- 💰 **Budget cuts directly reduce sales**  
  –10% across all channels → **–1.00 drop in sales**  

### ✅ Recommendations for Businesses
- Invest more in **TV and Radio advertising** (highest returns)  
- Reduce **Newspaper ad spend**, reallocate to TV/Radio  
- Use a **combined TV + Radio strategy** for maximum impact  
- Avoid **across-the-board budget cuts**  
- Follow a **data-driven marketing approach** with continuous monitoring  

---

## 🙌 Author

**Developed by Noor Fatima**  
🎓 Final-year Computer Science Student | Data Science Projects  
