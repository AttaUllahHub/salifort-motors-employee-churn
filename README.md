# Salifort Motors – Employee Churn Prediction & HR Analytics

## Project Overview

This project analyzes HR data from Salifort Motors to understand why employees leave the company and builds machine learning models that predict employee attrition. The goal is to turn raw HR data into actionable, data-driven recommendations that the HR department can use to improve employee retention.

## Business Problem

The HR department at Salifort Motors wants to improve employee satisfaction and retention, but doesn't know what to do with the employee data it has collected. The central business question is: **what's likely to make an employee leave the company?**

## Objective

The objective is to analyze the employee data and build a classification model that predicts whether an employee is likely to leave. Identifying at-risk employees in advance makes it possible to understand the factors driving attrition and intervene early — which is far less costly than finding, interviewing, and hiring replacements.

## Dataset

- **Source:** [HR Analytics and Job Prediction dataset](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction?select=HR_comma_sep.csv) (Kaggle)
- **Rows:** 14,999
- **Columns:** 10
- **Target variable:** `left` (1 = employee left, 0 = employee stayed)

| Feature | Description |
|---|---|
| `satisfaction_level` | Employee-reported job satisfaction level [0–1] |
| `last_evaluation` | Score of employee's last performance review [0–1] |
| `number_project` | Number of projects the employee contributes to |
| `average_monthly_hours` | Average number of hours worked per month |
| `time_spend_company` (renamed `tenure`) | Years at the company |
| `Work_accident` | Whether the employee had a workplace accident |
| `promotion_last_5years` | Whether the employee was promoted in the last 5 years |
| `Department` | Employee's department |
| `salary` | Employee's salary level (low / medium / high) |

## Project Workflow

1. **Data Loading** – Import the raw CSV into a pandas DataFrame.
2. **Data Cleaning** – Standardize column names, check for missing values, remove duplicate rows, and identify outliers in `tenure`.
3. **Exploratory Data Analysis** – Visualize relationships between hours worked, number of projects, satisfaction, evaluation scores, tenure, department, and attrition.
4. **Feature Engineering** – Encode categorical variables (`department`, `salary`) and engineer a binary `overworked` feature from `average_monthly_hours` to address potential data leakage.
5. **Model Building** – Fit a Logistic Regression model and, separately, Decision Tree / Random Forest models with cross-validated grid search for hyperparameter tuning.
6. **Model Evaluation** – Compare models using precision, recall, F1-score, accuracy, and AUC on held-out test data.
7. **Business Recommendations** – Translate model findings (feature importances, attrition patterns) into concrete HR actions.

## Exploratory Data Analysis

Key patterns identified in the notebook:

- Employees working on **7 projects all left** the company; those with 3–4 projects had the lowest attrition.
- There are two broad groups of employees who left: **overworked, high-performing employees** (~240–315 hrs/month) and **employees with more typical hours but lower satisfaction/evaluation scores**.
- Employees at **exactly the 4-year tenure mark** show unusually low satisfaction — a possible signal of a company-policy effect at that milestone.
- Employees with **more than 6 years of tenure rarely left** the company.
- No single department stood out with a disproportionately higher or lower attrition rate.
- A correlation heatmap confirmed that number of projects, monthly hours, and evaluation scores are positively correlated with each other, while satisfaction level is negatively correlated with attrition.

## Machine Learning

**Models used:** Logistic Regression, Decision Tree (`GridSearchCV`), Random Forest (`GridSearchCV`)

**Train/test methodology:** Stratified train/test split (75/25) to preserve the class balance of `left` (~83% stayed / 17% left); 4-fold cross-validation used during hyperparameter search for the tree-based models.

**Evaluation metrics:** Precision, Recall, F1-score, Accuracy, AUC

**Feature engineering for leakage control:** After an initial round of tree-based modeling, `satisfaction_level` was dropped and a new binary feature, `overworked` (average monthly hours > 175), replaced `average_monthly_hours`, to reduce the risk of data leakage and produce a model that generalizes better to a production setting.

### Results, as summarized in the notebook

| Model | Precision | Recall | F1-score | Accuracy | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 80%* | 83%* | 80%* | 83%* | — |
| Decision Tree (post feature engineering, as labeled in the notebook's summary) | 87.0% | 90.4% | 88.7% | 96.2% | 93.8% |

*Logistic Regression scores are weighted averages, as reported in the notebook's model summary.

### Actual computed test/cross-validation scores in the notebook

| Model | Precision | Recall | F1 | Accuracy | AUC |
|---|---|---|---|---|---|
| Decision Tree – Round 1 (CV) | 91.5% | 91.7% | 91.6% | 97.2% | 97.0% |
| Random Forest – Round 1 (CV) | 95.0% | 91.6% | 93.2% | 97.8% | 98.0% |
| Random Forest – Round 1 (test) | 96.4% | 92.0% | 94.1% | 98.1% | 95.6% |
| Decision Tree – Round 2, post feature engineering (CV) | 85.7% | 90.4% | 87.9% | 95.9% | 95.9% |
| Random Forest – Round 2, post feature engineering (CV) | 86.7% | 87.9% | 87.2% | 95.7% | 96.5% |
| Random Forest – Round 2, post feature engineering (test) | 87.0% | 90.4% | 88.7% | 96.2% | 93.8% |

The random forest model modestly outperformed the decision tree model at each stage. The Round 2 random forest test scores above are numerically identical to the figures the notebook's own "Summary of model results" cell attributes to the decision tree — see *Notes / Known Issues* below.

**Best-performing model:** Random Forest, Round 2 (trained without `satisfaction_level`, using the engineered `overworked` feature) — the notebook's designated final/champion model.

**Top predictive features** (both Decision Tree and Random Forest): `last_evaluation`, `number_project`, `tenure`, and `overworked`.

## Key Findings

- Employee attrition is strongly tied to being **overworked**: high project counts and long monthly hours are consistently associated with employees leaving.
- Employees who leave fall into two distinct groups — **overworked high performers** and **dissatisfied employees with more typical hours** — suggesting more than one retention lever is needed.
- Tenure has a non-linear relationship with attrition: risk appears to spike around the **4-year mark**, then drops sharply for employees who stay past 6 years.
- `last_evaluation`, `number_project`, `tenure`, and the engineered `overworked` feature are the most important predictors of attrition across both tree-based models.
- Removing `satisfaction_level` (a feature that may not be reliably available in production) and engineering `overworked` still resulted in a strong-performing model, addressing concerns about data leakage.

## Business Recommendations

- Cap the number of projects an employee can be assigned at one time.
- Investigate why employees around the four-year tenure mark are especially dissatisfied, and consider proactive promotion or recognition for this group.
- Reward employees fairly for working longer hours, or reduce the expectation that they do so.
- Clearly communicate overtime pay policies and workload/time-off expectations.
- Hold company-wide and team-level discussions to understand and address work culture.
- Avoid reserving high evaluation scores exclusively for employees working 200+ hours per month; use a more proportionate effort-to-reward scale.

**Suggested next steps:** Re-evaluate model performance with `last_evaluation` removed entirely, in case evaluation scores are not reliably available or are themselves influenced by attrition risk. A K-means clustering analysis on the same data could also surface additional employee segments worth investigating.

## Ethical Considerations

- **Employee privacy:** This dataset contains sensitive, individual-level HR information (performance evaluations, satisfaction, salary band). Any production use of a similar model should ensure data is anonymized, access-controlled, and used only for its stated purpose.
- **Fairness and bias:** A model that flags employees as "at risk of leaving" could be misused for pre-emptive negative action (e.g., reduced opportunities) against employees rather than for genuinely supportive interventions. Department-level EDA in this project did not reveal a specific department disproportionately affected, but this should be re-checked before any deployment, along with checks across other protected characteristics not present in this dataset.
- **Data leakage and reliability:** The project explicitly addresses a data leakage concern by removing `satisfaction_level` and engineering `overworked`. This reflects good practice — deploying a model on features unlikely to be available or reliable in production can produce misleadingly strong results that don't hold up in real-world use.
- **Human oversight:** Predictions should support, not replace, HR judgment and direct conversations with employees. False positives (see confusion matrix in the notebook) mean some employees would be incorrectly flagged as flight risks.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn (`LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GridSearchCV`, metrics)
- XGBoost (imported in the notebook; see *Notes* below — not used to fit or evaluate a model in this analysis)
- Pickle (Python standard library, used to save/load fitted models)

## Project Structure

```
salifort-motors-employee-churn/
│
├── Salifort_Motors_Employee_Churn.ipynb
├── dataset.csv
├── README.md
└── requirements.txt
```

## How to Run

1. Clone this repository.
2. (Recommended) Create and activate a virtual environment.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Ensure `dataset.csv` is in the project root (see *Notes* below — this file must be added manually).
5. Launch Jupyter and open the notebook:
   ```
   jupyter notebook Salifort_Motors_Employee_Churn.ipynb
   ```

## Notes / Known Issues

- **Metric-labeling inconsistency in the notebook's own summary:** The notebook's "Summary of model results" markdown cell attributes precision 87.0% / recall 90.4% / f1 88.7% / accuracy 96.2% / AUC 93.8% to "the decision tree model," but these figures are numerically identical to the actual computed Random Forest Round 2 test scores in the notebook (the decision tree's own test-set scores were never separately computed — only its cross-validation scores were). This is a pre-existing labeling error carried over from the original notebook and is reproduced above for transparency; the tables in the *Results* section show both the notebook's stated summary and the actual per-model figures computed in the code cells.
- **Dataset file not included:** The uploaded notebook loads a file named `HR_capstone_dataset.csv`, but no dataset file was provided alongside the notebook. You'll need to download it from the [Kaggle source](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction?select=HR_comma_sep.csv) (the Kaggle file is named `HR_comma_sep.csv`) and either rename it to `HR_capstone_dataset.csv` or update the `pd.read_csv(...)` call in the notebook. For this repository's structure, save it as `dataset.csv` and update the notebook's read path accordingly.
- **XGBoost is imported but unused:** The notebook's import cell imports `XGBClassifier`, `XGBRegressor`, and `plot_importance` from `xgboost`, but no XGBoost model is actually trained or evaluated anywhere in the analysis. It's kept in `requirements.txt` so the notebook's import cell runs without error, but you may want to remove the unused import from the notebook for a cleaner portfolio piece, or add an XGBoost model if you'd like to extend the comparison.
- **Model pickling path:** The notebook saves/loads fitted models using a local path (`/home/jovyan/work/`) from the original lab environment. Update this path if you re-run the notebook locally.
- **Package versions:** The exact versions of the libraries used to originally run this notebook aren't recorded in the file, so `requirements.txt` lists package names without pinned versions. Pin versions once you've confirmed the notebook runs cleanly in your environment.

## Author

**Atta Ullah**
BS Data Science Student
GitHub: [AttaUllahHub](https://github.com/AttaUllahHub)
LinkedIn: [https://www.linkedin.com/in/attaullah57]
