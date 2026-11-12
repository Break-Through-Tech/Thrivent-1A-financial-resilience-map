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

 - Developed stratified 80/20 training/test preprocessing workflow and trained a logistic regression, gradient-boosting classifier, and random forest model.

 - Identified 446 exploratory candidate features during initial preprocessing and refined the final modeling dataset to 425 predictors after additional exclusions to reduce target leakage and remove survey administration variables.

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

For those who want to train the recommended model or generate predictions locally without working through the notebooks, we have also added standalone Python scripts in `src/`. Instructions for that workflow are provided below.

**Requirements:** Python 3.10 and the packages listed in `requirements.txt`.

#### 1. Install dependencies

From the repository root, create and activate a virtual environment using the instructions above, then run:

```bash
python -m pip install -r requirements.txt
```

#### 2. Train the model

```bash
python -m src.train
```

The training script uses the existing 2025 SHED modeling dataset and feature specifications. It follows the methodology in `04_logistic_regression_baseline.ipynb`, including:

- Our 80/20 train-test split
- Five-fold cross-validation
- Population survey weights
- Selection of the regularization strength using weighted log loss

The fitted model and its metadata are saved as:

- `models/logistic_regression_with_ef.joblib`
- `models/logistic_regression_with_ef.json`

The `.joblib` file is generated locally and excluded from Git. Users must train the model before running predictions.

#### 3. Generate predictions

```bash
python -m src.predict --input data/processed/model_dataset_2025.csv --output outputs/predictions.csv
```

The prediction script loads the fitted Logistic Regression pipeline, validates the required input features, and exports the results to a CSV file.

The output contains:

| Column | Description |
|---|---|
| `shedid` | Respondent identifier, if available |
| `fragility_probability` | Predicted probability of financial fragility |
| `predicted_fragile` | Binary classification using a 0.5 threshold |

A classification of `1` indicates predicted financial fragility, while `0` indicates predicted non-fragility.

The input CSV must contain the features listed in the saved model metadata. The supplied modeling dataset can be used to demonstrate the inference workflow, although predictions on training observations should not be used as independent performance estimates.

#### 4. Run the prediction app

```bash
streamlit run app.py
```

The app asks 8 questions and shows the estimated probability of financial fragility and the resulting classification. All 8 questions must be answered before the app makes a prediction. The sidebar can fill the form with one of three sample profiles or clear it.

You don't need to train the model first. If `models/logistic_regression_with_ef.joblib` is missing, the app rebuilds it on first launch (this takes a few seconds).

For how the questions were chosen and how 8 answers become a full model input, see [Prediction App: Design Decisions](#prediction-app-design-decisions) below.

#### 5. Run the tests

```bash
pytest
```

`tests/test_inference.py` contains 14 tests. They run three representative profiles (financially secure, getting by, and financially stretched) through the full inference path and check the averaging over similar respondents, partial answers, linked survey features, and input validation.


#### 6. Continuous Integration

This project uses **GitHub Actions** to automatically check the code when changes are pushed to the `main` branch or when a pull request targets `main`.

The CI workflow:
- Installs project dependencies.
- Runs code-quality checks using Flake8.
- Runs automated tests using pytest when test files are available.

The workflow is located at `.github/workflows/ci.yml`.

### Project structure

```
Thrivent-1A-financial-resilience-map/
├── README.md
├── Challenge-Project-Overview.md
├── Getting-Started-for-Fellows.md
├── requirements.txt
├── pytest.ini
├── app.py                      # Streamlit prediction app
├── database.py                 # SQLite prediction storage
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
│   ├── 08_success_metrics.ipynb
│   ├── 09_segmentation.ipynb
│   └── 10_final_model_comparison.ipynb
│
├── models/
│   ├── README.md
│   ├── logistic_regression_with_ef.json
│   ├── gradient_boosting_with_ef.json
│   └── gradient_boosting_without_ef.json
│
├── src/
│   ├── model.py                # load the saved model and predict
│   ├── train.py                # train the logistic regression model
│   ├── predict.py              # batch predictions from a CSV
│   └── inference.py            # turn the app's 8 answers into a prediction
│
├── tests/
│   └── test_inference.py
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

## Prediction Database: Storing the Model Predictions

`database.py` uses SQLite to store predictions in
`financial_predictions.db`. No additional packages are required.

Each record (saved prediction) contains:
- A unique record ID
- A prediction: `0` = not financially fragile, `1` = financially fragile
- An automatically generated timestamp in UTC (Coordinated Universal Time)

### Create the database

Run the following command from the main project folder:

```bash
python -c "from database import initialize_database; initialize_database()"
```

### Save and retrieve predictions

In Python, use:

```python
from database import initialize_database, save_prediction, get_predictions

initialize_database()

# Store an example prediction and return its record ID
record_id = save_prediction(1)

# Retrieve saved predictions, in descending order (newest first)
records = get_predictions()
print(records)
```

`save_prediction()` rejects values other than 0 or 1.
Each successful call adds a new record to the database.

The database is created locally and ignored by Git.

---

## Prediction App: Design Decisions

*Question selection, inference method, and validation by Ansa Kanwal (November Task 2). Form interface by Ethan Wang (November Task 1), built on `src/inference.py`.*

The model was trained on 425 survey features, but no user can be expected to answer 425 questions. The app asks **8**. This section explains which questions were chosen and why, how 8 answers become a full model input, how we checked the result, and how it relates to the Task 5 segmentation. The code is in `src/inference.py`.

### Results at a glance

All 2,587 held-out test respondents, weighted to the U.S. adult population, at the 0.5 threshold:

| | Full model (all 425 answers) | App (8 answers) |
|---|---|---|
| **Recall** (share of fragile people caught) | 78.0% | **80.2%** |
| Precision (share of flagged people who are fragile) | 82.1% | 79.8% |
| ROC AUC | 0.925 | 0.923 |
| Brier score (lower is better) | 0.103 | 0.105 |
| Accuracy | 85.6% | 85.2% |
| Same classification as the full model | | 92.7% |

With only 8 answers, the app catches slightly more fragile respondents than the full model (182 missed in the test set vs 210), and its ranking quality is close to the full model's (ROC AUC 0.923 vs 0.925). The trade-off is a few more false alarms.

### How a prediction is made

1. The user answers the 8 questions.
2. `src/inference.py` finds the 100 training respondents whose answers are most similar to the user's.
3. Each of those 100 respondents is copied, and their answers to the 8 questions are replaced with the user's. Their answers to every other question stay as they are.
4. All 100 copies go through the saved preprocessing and model pipeline (`src/model.py`) unchanged.
5. The app shows the **average** of the 100 probabilities and classifies it at the model's 0.5 threshold.

### Why these 8 questions

**The 8 questions cover all of the model's top 10 features.** They were chosen from the Task 3 SHAP analysis (`notebooks/08_success_metrics.ipynb`), which ranks how much each survey question drives the logistic regression's predictions (`outputs/task3/shap_importance.csv`). We re-computed that ranking independently and it matched exactly.

| Question in the app | Survey feature (SHAP rank) |
|---|---|
| In the past 12 months, how often did you carry an unpaid credit card balance? | `C4A` (#1) |
| Do you have emergency savings that would cover 3 months of expenses? | `EF1` (#2), `EF2` (#3) |
| How often do you have money left over at the end of the month? | `B1_a` (#4) |
| Do you have a savings account, money market account, or CD? | `K21_d` (#5) |
| In the past month, which best describes the food eaten in your household? | `FD3` (#6) |
| What is the largest emergency expense you could cover using only your savings? | `EF7` (#7), `EF7A` (#9) |
| In the past month, was your total spending less than, the same as, or more than your income? | `I20` (#8) |
| How well are you managing financially these days? | `B2` (#10) |

Some questions set two features:

- The emergency savings answer also sets `EF2`, the follow-up asked only of people without savings ("could you cover 3 months by borrowing, using savings, or selling assets?").
- `EF7` and `EF7A` are the same question, asked to two halves of the survey with different top dollar ranges. One question covers both.

**Wording.** Every question was checked against the [SHED 2025 codebook](https://www.federalreserve.gov/consumerscommunities/files/SHED_2025codebook.pdf).

**No demographic questions.** Earlier versions also asked six household questions (income, employment, education, housing, age, and household size), on the idea that they would help find similar respondents. We tested that with 5-fold cross-validation on the training split (100 similar respondents; the test set was not used):

| Questions asked | Recall | ROC AUC | Log loss |
|---|---|---|---|
| 8 SHAP questions + all 6 household questions (14) | 79.9% | 0.927 | 0.331 |
| 8 SHAP questions + income + age (10) | 80.6% | 0.927 | 0.332 |
| 8 SHAP questions + income (9) | 79.2% | 0.927 | 0.332 |
| **8 SHAP questions only (used)** | **79.7%** | **0.926** | **0.333** |

The household questions add almost nothing (ROC AUC changes by 0.001, and recall differences of about ±1.5 points are within noise for this sample), so they were removed. This makes the form shorter and more approachable, and the app no longer asks any demographic questions. Race, ethnicity, and gender were never asked; see [Limitations](#limitations) for how they still enter the model.

**Why `B1_a` is included.** The first version left out `B1_a` (SHAP #4) because its wording had not been confirmed. The codebook confirmed it, and in cross-validation it gave a small but consistent improvement at every setting tried (log loss 0.001 to 0.004 lower, precision +0.7 to +1.1 points, recall +0.2 to +1.0 points, compared with leaving it out).

### How the unasked answers are filled in

We tried three methods. Each was compared on how well it catches financially fragile respondents, the project's main goal.

| Method | Result | Decision |
|---|---|---|
| 1. Fill every unasked answer with the most common answer overall | The most common answers describe a financially secure person. A clearly high-risk profile scored 64%, while real test respondents with similar answers averaged 93%. | Rejected |
| 2. Find the 100 most similar respondents and build **one** profile from their most common answers | On the full test set it caught only **68%** of fragile respondents, vs 78% for the full model. A single "most typical" profile pulls every prediction toward the middle, so borderline fragile people fall under 50%. | Rejected |
| 3. Score **each** of the 100 similar respondents with the user's answers and **average** | Catches **80.2%** of fragile respondents (table above). | **Used** |

**Why method 3 is sound.** It combines two standard survey-statistics methods:

- **Nearest-neighbour ("hot-deck") imputation**, which fills missing answers from similar real respondents. Survey agencies such as the U.S. Census Bureau have used it for decades (Andridge & Little, 2010).
- **Averaging predictions over several plausible fills**, the core idea of multiple imputation (Rubin, 1987). Each similar respondent is one realistic, internally consistent way the unasked answers could look. Averaging keeps that uncertainty instead of replacing it with a single guess.

**Rare answers count for more.** When finding similar respondents, each matching answer is weighted by how rare it is (`-log` of the share of respondents who gave it), an inverse-frequency similarity measure for categorical data (Boriah, Chandola & Kumar, 2008). For example, only about 2% of respondents often don't have enough to eat, while about 61% have a savings account. A user giving both answers is matched mainly on the rare one and scores 88%.

**Linked features stay consistent.** Some features are derived from others, so the app sets them together. For example, `atleast_okay` follows `B2`, and answering "I don't have a credit card" also sets `C2A` (has a credit card) to "No".

**Why 100 similar respondents.** We tested 10, 25, 50, 100, 200, 500, and 1,000 with 5-fold cross-validation on the training split. Each fold's model was refit without that fold, and similar respondents were drawn only from the other folds. The test set was not used. With the 14-question version, results were nearly identical across the whole range:

| Similar respondents | 10 | 25 | 50 | 100 | 200 | 500 | 1,000 |
|---|---|---|---|---|---|---|---|
| 14 questions: log loss | 0.332 | 0.332 | 0.331 | 0.331 | 0.331 | 0.332 | 0.333 |
| 14 questions: recall | 79.5% | 80.0% | 79.8% | 79.9% | 79.7% | 79.9% | 79.2% |
| 14 questions: ROC AUC | 0.927 | 0.927 | 0.927 | 0.927 | 0.928 | 0.927 | 0.928 |

100 sits in the middle of this flat range, so the result does not depend on a lucky setting. The 8-question set was checked at 100 (table above).

### How we checked it

- **Full test set:** the results table at the top of this section. All 2,587 held-out respondents were scored once with all 425 of their real answers and once with only their 8 app answers.
- **Sample profiles** (loadable from the app's sidebar). Each matches the real outcomes of the 100 survey respondents the app finds for it:

  | Sample profile | App probability | Classification | Actual fragile rate of its 100 similar respondents |
  |---|---|---|---|
  | Financially secure | 1.6% | Not fragile | 0.9% |
  | Getting by | 63.2% | Fragile | 65.9% |
  | Financially stretched | 97.5% | Fragile | 98.3% |

- **Automated tests:** `tests/test_inference.py` runs these profiles and other checks on every pull request through GitHub Actions.
- **Task 3 outputs verified:** we independently recomputed the main metrics in `outputs/task3/metrics.csv` (ROC AUC, PR AUC, Brier, log loss, accuracy, precision, recall) from the saved test predictions, and the logistic regression SHAP ranking. Both matched.
- **Caveat:** several methods were compared on the same test set while developing the app. The question set and the number of similar respondents were chosen by cross-validation on the training data only, and the improvement from averaging (68% to 80.2% recall) is far larger than test-set noise.

### How this relates to the segmentation (Task 5)

**Did the segmentation shape the question choice?** No. The questions come from the SHAP ranking. The two analyses do point at some of the same variables: 4 of the 8 questions (`B2`, `FD3`, `K21_d`, and `C2A` through the credit card question) are also features the Task 5 segments are built on.

**Could the app tell users which household segment they are in?** Not reliably with the current 8 questions:

- Assigning respondents to the nearest saved segment profile (`outputs/task5/segment_profiles.csv`) reproduces the notebook 09 segments for 99.1% of all 12,934 respondents, so assignment itself works.
- But using only the 8 app answers (assigning each of the 100 rows the app builds for a respondent, then taking the most common segment), only 47.4% of test respondents land in the same segment as in notebook 09. The segments depend heavily on age and household size (age has Cramér's V 0.55 with segment), which the app no longer asks. With the earlier 14-question version, which asked them, agreement was 93.7%.
- So showing segments would mean adding the age and household questions back. They don't improve the fragility prediction (see above).

**We also recommend caution before showing segments to users.** The Task 5 stability check found that only segment 1 (older, comfortable homeowners) is stable across refits (mean Jaccard 0.88). Segments 2 to 4 average 0.63 to 0.68, so their boundaries depend on the sample. If segments are added, they should be presented as rough groupings, not fixed types. This is not implemented yet and is a decision for the team.

### Other decisions

- **Model:** the app uses `logistic_regression_with_ef`, the model recommended in Task 4 and produced by `src/train.py`. The inference code reuses `src/model.py` rather than duplicating its loading and prediction logic.
- **No model file needed to deploy:** `.joblib` files are git-ignored. If the file is missing, the app rebuilds the model once using the regularization strength saved in `models/logistic_regression_with_ef.json`. This is the same final fit as `src/train.py` without repeating cross-validation, so the app works on a fresh clone or on Streamlit Community Cloud.
- **Interface:** Task 2 included a basic placeholder `app.py`, which was always meant to be replaced. Ethan's Task 1 form replaced it. It builds its questions from `INPUT_FIELDS` and calls `FragilityPredictor.from_artifacts()` and `predictor.predict(answers)`.
- **All questions required:** `src/inference.py` can handle skipped questions, but the form requires all 8 answers so that every prediction uses the full set.

### Limitations

- The probability is a statistical estimate from survey data. It is not financial advice or an assessment of any individual.
- To catch more fragile people, the app raises slightly more false alarms than the full model (precision 79.8% vs 82.1%).
- The app asks no demographic questions, but race, ethnicity, gender, age, and income remain model inputs. They are taken from the similar respondents, so they can still influence predictions indirectly (all 8 protected attributes together are 3.4% of the logistic regression's SHAP importance). Whether to keep protected attributes in the model is an open team decision (see notebook 10).
- Similar respondents are matched without survey weights.
- The saved model must be loaded with scikit-learn 1.7.2, the version pinned in `requirements.txt`.

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

07_model_evaluation.ipynb provides additional evaluation of the logistic regression baseline, while 08_success_metrics.ipynb consolidates comparisons across all three approaches. 10_final_model_comparison.ipynb combines the two into the final comparison, adding confusion matrices for all three models, a threshold trade-off, a subgroup check, and a model recommendation.

The shared evaluation includes ROC AUC, PR AUC, Brier score, log loss, classification metrics, calibration analysis, bootstrap confidence intervals, and SHAP-based feature interpretation.

Detailed model performance and findings are presented in the Results & Key Findings section below.

---

## 📈 **Results & Key Findings**
We evaluated Logistic Regression, Gradient Boosting, and Random Forest using the same held-out test set of 2,587 respondents. All reported metrics are population-weighted using weight_pop, with classification metrics calculated at a probability threshold of 0.5.

**Model Performance Comparison**

| Model (with EF) | ROC AUC | PR AUC | Brier Score ↓ | Accuracy | Recall |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.925 | 0.875 | 0.103 | 85.6% | 78.0% |
| Gradient Boosting | **0.928** | **0.880** | **0.100** | **86.2%** | **79.2%** |
| Random Forest | 0.919 | 0.869 | 0.111 | 85.1% | 76.7% |
| Training Prevalence Baseline | 0.500 | 0.370 | 0.233 | 63.0% | 0.0% |

All three trained models substantially outperform the reference baseline. Gradient Boosting achieves the strongest predictive performance, closely followed by Logistic Regression.

**Key Findings**

1. *Gradient Boosting and Logistic Regression perform comparably.*

Gradient Boosting achieves the highest ROC AUC (0.928) and lowest Brier score (0.100). However, paired bootstrap confidence intervals for the differences between Gradient Boosting and Logistic Regression include zero, indicating that their observed performance differences are not clearly distinguishable given the held-out sample. Random Forest performs slightly worse on discrimination and probability accuracy, with a ROC AUC of 0.919 and Brier score of 0.111.

2. *The models identify most financially fragile respondents.*

At a classification threshold of 0.5, all three models achieve approximately 77–79% recall, correctly identifying more than three-quarters of financially fragile respondents. Precision ranges from approximately 82% to 83%, indicating that most respondents predicted to be financially fragile belong to the positive class.

3. *Removing emergency-finance questions reduces performance, but there still remains a predictive signal.*

We conducted a sensitivity analysis excluding 16 emergency-finance (EF) predictors closely related to the outcome variable. Removing these features reduces ROC AUC by approximately 0.011–0.012 across the three models. However, all models maintain ROC AUC scores above 0.90, suggesting that broader household, financial, and demographic characteristics also contain information associated with financial fragility.


4. *Feature importance reveals several consistent predictors.*

We used SHAP to examine which survey questions contribute most strongly to model predictions. The models consistently identify C4A, EF1, EF2, K21_d, and FD3 among their ten highest-ranked predictors. While the models share several important predictors, their complete importance rankings differ. Gradient Boosting relies more on race and ethnicity (`race_5cat` ranks 20th of 425 questions, vs 44th for Logistic Regression and 57th for Random Forest).

5. *Performance differs across demographic groups.*

All three models rank Black respondents less accurately (ROC AUC about 0.85, vs 0.92–0.93 for White respondents) and understate their risk. They also miss more fragile respondents in groups where fragility is less common, such as incomes of $100,000 or more and ages 75+. Because every model shows these patterns, they reflect the data rather than one model's design.

6. *Recommended model: Logistic Regression.*

Logistic Regression performs as well as Gradient Boosting within test-set noise, is the best calibrated, and can be explained directly through its coefficients. It also relies less on race and ethnicity and on the emergency-finance questions. Gradient Boosting is kept as a cross-check, and Random Forest is not carried forward. See 10_final_model_comparison.ipynb for the full comparison.

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

## 📄 **References**

Board of Governors of the Federal Reserve System. Survey of Household Economics and Decisionmaking (SHED). DOI: 10.17016/datasets.002

---

## 🙏 **Acknowledgements** 

This project is part of the Break Through Tech AI Studio – Fall 2026 program.

**Host Organization:** Thrivent
**Challenge Advisor:** Nathan Rickert
**AI Studio Coach:** Julio Contreras

We thank our Challenge Advisor, AI Studio Coach, Break Through Tech staff, and team members for their guidance and collaboration throughout the project.

We also acknowledge the Board of Governors of the Federal Reserve System for making the Survey of Household Economics and Decisionmaking public-use microdata, questionnaires, codebooks, and published response tables available for research and educational use.
