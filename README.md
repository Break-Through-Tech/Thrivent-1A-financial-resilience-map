# Thrivent1A Financial Resilience Map

### 👥 **Team Members**

| Name           | GitHub Handle   | Contribution                                                                        |
| -------------- | --------------- | ----------------------------------------------------------------------------------- |
| Lynn Aung      | @loofsan        | Model training, feature engineering, hyperparameter tuning and performance analysis |
| Valerie Weiner | @valerieanne12  | Data collection, exploratory data analysis (EDA), dataset documentation             |
| Ansa Kanwal    | @kansa05        | Model selection, hyperparameter tuning, model training and optimization             |
| Sarah Wheeler  | @sarahaawheeler | Model building, performance analysis, results interpretation                        |
| Yasirah Waites | @Ywaites        | Feature Engineering, Data exploration, Model training                               |
| Ethan Wang     | @etwa5465       | Data preprocessing, feature engineering, visualization                              |
| Pirm Dhararag  | @Pirm-Dhararag  | EDA, model training, model explainability                                           |

---

## 🎯 **Project Highlights**

The Financial Resilience Map investigates household financial fragility using the Federal Reserve's 2025 Survey of Household Economics and Decisionmaking (SHED).

Completed work:

 - Replicated the Federal Reserve's published estimate that 63% of U.S. adults could cover a $400 emergency expense using cash or its equivalent. Our survey-weighted estimate was 63.14%, a difference of 0.14 percentage points.

 - Constructed a binary financial-fragility outcome from the pay_casheqv survey variable.

 - Prepared a feature matrix with 446 predictors and developed stratified 80/20 training/test preprocessing workflow.

**Model development:** See the model results and artifacts sections for the latest training, validation, and evaluation status.
---

## 👩🏽‍💻 **Setup and Installation**

**IDE Used:** Google Colab

To get started with this project, first clone this repository to your computer with the following command:

```bash
git clone https://github.com/Break-Through-Tech/Thrivent-1A-financial-resilience-map.git
cd Thrivent-1A-financial-resilience-map
```

Then, install dependencies via the requirements.txt:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.\.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### Project structure

```
Thrivent-1A-financial-resilience-map/
├── README.md
├── Challenge-Project-Overview.md
├── Getting-Started-for-Fellows.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── public2025.csv
│   └── processed/
│       ├── public2025_clean.csv
│       ├── model_dataset_2025.csv
│       ├── model_dataset_2025_spec.json
│       └── logistic_test_predictions.csv
│
├── notebooks/
│   ├── 01_clean_survey.ipynb
│   ├── 02_replicate_fed_estimate.ipynb
│   ├── 03_build_model_dataset.ipynb
│   ├── 04_logistic_regression_baseline.ipynb
│   ├── 05_gradient_boosting.ipynb
│   ├── 06_tree_based_model.ipynb
│   ├── 07_model_evaluation.ipynb
│   └── 08_success_metrics.ipynb
│
├── models/
│   ├── README.md
│   ├── gradient_boosting_with_ef.json
│   └── gradient_boosting_without_ef.json
│
└── outputs/
    └── task3/
        ├── bootstrap_intervals.csv
        ├── bootstrap_paired_differences.csv
        ├── calibration_bins.csv
        ├── metrics.csv
        ├── shap_importance.csv
        └── test_predictions.csv
```

---

## 🏗️ **Project Overview**

The Financial Resilience Map is a Fall 2026 Break Through Tech AI Studio challenge hosted in collaboration with Thrivent.

The project investigates a question relevant to financial services, community lenders, credit unions, financial-education organizations, and public agencies: Which households are financially fragile, and what factors are associated with that fragility?

Traditional measures such as credit scores primarily describe a person's history of repaying debt. Financial resilience captures a different dimension: whether a household can absorb an unexpected financial shock, such as a car repair, reduction in working hours, or medical expense.

The Federal Reserve's Survey of Household Economics and Decisionmaking provides a direct way to study this question. One of its headline measures asks whether adults could cover a $400 emergency expense using cash or its equivalent. In the 2025 survey, 63% of adults reported being able to do so.

This project moves beyond this population-level statistic by developing machine-learning models that estimate household financial fragility and explain the factors associated with each prediction.

Our intended deliverable is decision support for humans, particularly for financial education and informed conversations. It is not intended to automate financial decisions or determine an individual's eligibility, creditworthiness, or access to financial products.

## 📊 **Data Exploration**

We use the **2025 Survey of Household Economics and Decisionmaking (SHED)** dataset. The raw dataset contains **12,934 respondents and 815 columns**. Respondent IDs (`shedid`) and the four supplied survey weights (`weight`, `weight_pop`, `panel_weight`, and `panel_weight_pop`) were validated before preprocessing.

### Data Exploration and Preprocessing

- Reviewed missing values, variable types, and survey responses.
- Identified and excluded 357 imputation flag (\_iflag) columns from the model features.
- Kept all 12,934 respondents instead of removing rows with missing values.
- Preserved survey skip responses as `__NOT_ASKED__`.
- Identified **446 candidate features**:
  - 23 numeric
  - 421 categorical
  - 2 ordinal
- Filled missing values and converted categorical and ordinal responses into a format suitable for modeling.
- Used weight_pop to calculate population-level distributions.
- Checked for potential outliers without automatically removing them.
- Exported a cleaned version of the dataset for future analysis.

### Key Considerations

Survey blanks can represent questions that were not asked rather than missing responses. We therefore kept these responses separate from regular missing values. Survey weights were also preserved for population-level analysis.

---

## 🧠 **Model Development**

To ensure consistent comparisons across models, we created a shared modeling dataset in 03_build_model_dataset.ipynb. This dataset contains 12,934 respondents, a binary financial-fragility target, survey population weights, and predefined training, testing, and cross-validation assignments.

The target variable, financially_fragile, is derived from pay_casheqv, indicating whether a respondent could cover a $400 emergency expense using cash or its equivalent. Respondents who answered "No" are labeled financially fragile (1), while those who answered "Yes" are labeled not financially fragile (0).

Feature selection excludes respondent identifiers, survey weights, imputation flags, survey administration metadata, state-level geography, and the EF3_* variables directly involved in constructing the target to reduce data leakage.

The resulting dataset contains 425 predictors: 11 numerical, 412 categorical, and 2 ordinal features. An additional feature set containing 409 predictors excludes 16 emergency-finance (EF) variables to examine how much model performance depends on questions closely related to the prediction target.
Training and Preprocessing
We use a shared 80/20 stratified training/test split, containing 10,347 training respondents and 2,587 testing respondents. Five predefined stratified cross-validation folds are used for hyperparameter selection.

Preprocessing is implemented within each model's scikit-learn pipeline to avoid fitting transformations on held-out observations.
- Numerical features: Median imputation with missingness indicators. Numerical features are standardized for logistic regression.
- Categorical features: Missing-value handling and one-hot encoding, with unknown categories ignored during prediction.
- Ordinal features: Encoding based on predefined category orderings, with model-specific scaling as appropriate.
Models are trained using population survey weights (weight_pop) normalized to a mean of one. Original survey weights are retained for population-weighted performance evaluation.
Models Implemented
We developed and compared three supervised classification models:
Model	Notebook	Approach
Logistic Regression	04_logistic_regression_baseline.ipynb	Regularized linear baseline with five-fold cross-validation for selecting regularization strength
Gradient Boosting	05_gradient_boosting.ipynb	Sequential tree-based ensemble with grid search over learning rate, number of estimators, tree depth, and subsampling
Random Forest	06_tree_based_model.ipynb	Bagged decision-tree ensemble with cross-validation over tree complexity and feature-sampling settingsHyperparameters are selected using population-weighted cross-validation log loss. Each model is fitted on the training partition and evaluated against the same held-out test respondents.
Sensitivity Analysis and Model Comparison
Each modeling approach is evaluated with and without the 16 emergency-finance variables. This sensitivity analysis investigates whether performance is driven primarily by questions closely related to the target or whether broader household, demographic, and financial characteristics also provide predictive information.

07_model_evaluation.ipynb provides additional evaluation of the logistic regression baseline, while 08_success_metrics.ipynb consolidates comparisons across all three approaches.

The shared evaluation includes ROC AUC, PR AUC, Brier score, log loss, classification metrics, calibration analysis, bootstrap confidence intervals, and SHAP-based feature interpretation.

Detailed model performance and findings are presented in the Results & Key Findings section below.

---

## 📈 **Results & Key Findings**
We evaluated Logistic Regression, Gradient Boosting, and Random Forest using the same held-out test set of 2,587 respondents. All reported metrics are population-weighted using weight_pop, with classification metrics calculated at a probability threshold of 0.5.

**Model Performance Comparison**

| Model (with EF) | ROC AUC | PR AUC | Brier Score ↓ | Accuracy | Recall |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.925 | 0.875 | 0.103 | 85.6% | 78.0% |
| Gradient Boosting | **0.929** | **0.881** | **0.101** | **85.8%** | **78.2%** |
| Random Forest | 0.919 | 0.869 | 0.111 | 85.1% | 76.7% |
| Training Prevalence Baseline | 0.500 | 0.370 | 0.233 | 63.0% | 0.0% |

All three trained models substantially outperform the reference baseline. Gradient Boosting achieves the strongest predictive performance, closely followed by Logistic Regression.

**Key Findings**

1. *Gradient Boosting and Logistic Regression perform comparably.*

Gradient Boosting achieves the highest ROC AUC (0.929) and lowest Brier score (0.101). However, paired bootstrap confidence intervals for the differences between Gradient Boosting and Logistic Regression include zero, indicating that their observed performance differences are not clearly distinguishable given the held-out sample. Random Forest performs slightly worse on discrimination and probability accuracy, with a ROC AUC of 0.919 and Brier score of 0.111.

2. *The models identify most financially fragile respondents.*

At a classification threshold of 0.5, all three models achieve approximately 77–78% recall, correctly identifying more than three-quarters of financially fragile respondents. Precision ranges from approximately 82% to 83%, indicating that most respondents predicted to be financially fragile belong to the positive class.

3. *Removing emergency-finance questions reduces performance, but there still remains a predictive signal.*

We conducted a sensitivity analysis excluding 16 emergency-finance (EF) predictors closely related to the outcome variable. Removing these features reduces ROC AUC by approximately 0.011–0.013 across the three models. However, all models maintain ROC AUC scores above 0.90, suggesting that broader household, financial, and demographic characteristics also contain information associated with financial fragility.


4. *Feature importance reveals several consistent predictors.*

We used SHAP to examine which survey questions contribute most strongly to model predictions. The models consistently identify C4A, EF1, EF2, K21_d, FD3, and B2 among their ten highest-ranked predictors. While the models share several important predictors, their complete importance rankings differ. 

---

## 🚀 **Next Steps**

**You might consider addressing the following (as applicable):**

- What are some of the limitations of your model?
- What would you do differently with more time/resources?
- What additional datasets or techniques would you explore?

---

## 📝 **License**

This project is licensed under the MIT License.

---

## 📄 **References** (Optional but encouraged)

Board of Governors of the Federal Reserve System. Survey of Household Economics and Decisionmaking (SHED). DOI: 10.17016/datasets.002

---

## 🙏 **Acknowledgements** (Optional but encouraged)

This project is part of the Break Through Tech AI Studio – Fall 2026 program.

**Host Organization:** Thrivent
**Challenge Advisor:** Nathan Rickert
**AI Studio Coach:** Julio Contreras

We thank our Challenge Advisor, AI Studio Coach, Break Through Tech staff, and team members for their guidance and collaboration throughout the project.

We also acknowledge the Board of Governors of the Federal Reserve System for making the Survey of Household Economics and Decisionmaking public-use microdata, questionnaires, codebooks, and published response tables available for research and educational use.
