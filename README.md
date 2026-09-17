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

---

## 🎯 **Project Highlights**

**Example:**

- Developed a machine learning model using `[model type/technique]` to address `[challenge project task]`.
- Achieved `[key metric or result]`, demonstrating `[value or impact]` for `[host company]`.
- Generated actionable insights to inform business decisions at `[host company or stakeholders]`.
- Implemented `[specific methodology]` to address industry constraints or expectations.

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

Relevant datasets can be found in the /data folder. The notebook should automatically point to this dataset, so ensure that the data is present in this folder. The dataset is titled "public2025.csv."

Load the notebook in your preferred IDE (we used Colab) and run all cells.

---

## 🏗️ **Project Overview**

The Financial Resilience Map is a Fall 2026 Break Through Tech AI Studio challenge hosted in collaboration with Thrivent.

The project investigates a question relevant to financial services, community lenders, credit unions, financial-education organizations, and public agencies: Which households are financially fragile, and what factors are associated with that fragility?

Traditional measures such as credit scores primarily describe a person's history of repaying debt. Financial resilience captures a different dimension: whether a household can absorb an unexpected financial shock, such as a car repair, reduction in working hours, or medical expense.

The Federal Reserve's Survey of Household Economics and Decisionmaking provides a direct way to study this question. One of its headline measures asks whether adults could cover a $400 emergency expense using cash or its equivalent. In the 2025 survey, 63% of adults reported being able to do so.

This project moves beyond this population-level statistic by developing machine-learning models that estimate household financial fragility and explain the factors associated with each prediction.

Our intended deliverable is decision support for humans, particularly for financial education and informed conversations. It is not intended to automate financial decisions or determine an individual's eligibility, creditworthiness, or access to financial products.

## 📊 **Data Exploration**

**You might consider describing the following (as applicable):**

- The dataset(s) used: origin, format, size, type of data
- Data exploration and preprocessing approaches
- Insights from your Exploratory Data Analysis (EDA)
- Challenges and assumptions when working with the dataset(s)

**Potential visualizations to include:**

- Plots, charts, heatmaps, feature visualizations, sample dataset images

---

## 🧠 **Model Development**

**You might consider describing the following (as applicable):**

- Model(s) used (e.g., CNN with transfer learning, regression models)
- Feature selection and Hyperparameter tuning strategies
- Training setup (e.g., % of data for training/validation, evaluation metric, baseline performance)

---

## 📈 **Results & Key Findings**

**You might consider describing the following (as applicable):**

- Performance metrics (e.g., Accuracy, F1 score, RMSE)
- How your model performed
- Insights from evaluating model fairness

**Potential visualizations to include:**

- Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

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
