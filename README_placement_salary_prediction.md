# Placement Salary Prediction

A machine learning project that uses student academic performance, educational background, work experience, specialization, and placement status to predict placement salary using **Linear Regression**.

> **Dataset note:** The dataset used in this project was provided externally and is not included in this repository. The notebook currently loads `Placement_data.csv` from a local file path.

---

## 📌 Project Overview

This project explores a placement dataset containing academic, demographic, educational, work-experience, and placement-related information.

The main objective is to build a **regression model that predicts salary** based on the available student and placement features.

The notebook follows an end-to-end workflow:

```text
Load Dataset
     ↓
Understand Dataset
     ↓
Check Missing Values
     ↓
Handle Missing Salary Values
     ↓
Encode Categorical Variables
     ↓
Detect Outliers
     ↓
Winsorization
     ↓
Feature Scaling
     ↓
Train/Test Split
     ↓
Linear Regression
     ↓
Predictions
     ↓
Model Evaluation
```

---

## 🎯 Objective

The primary objective is:

> **To predict the salary of a student using academic performance, educational background, work experience, specialization, and placement-related features.**

The target variable is:

```text
salary
```

---

## 📊 Dataset

The notebook works with **215 records and 14 columns**.

The dataset contains the following variables:

| Column | Description |
|---|---|
| `gender` | Student gender |
| `ssc_p` | Secondary School Certificate percentage |
| `ssc_b` | SSC board |
| `hsc_p` | Higher Secondary percentage |
| `hsc_b` | HSC board |
| `hsc_s` | HSC specialization/stream |
| `degree_p` | Degree percentage |
| `degree_t` | Degree type |
| `workex` | Whether the student has work experience |
| `etest_p` | Employability test percentage |
| `specialisation` | MBA specialization |
| `mba_p` | MBA percentage |
| `status` | Placement status |
| `salary` | Placement salary — target variable |

The notebook output confirms that the original dataset contains **215 rows and 14 columns**, with 148 non-null salary values and 67 missing salary values. fileciteturn4file0

---

## 🔍 Exploratory Data Analysis

The notebook begins by loading the dataset and inspecting its structure.

### Dataset Information

The `df.info()` check shows:

- **215 observations**
- **14 columns**
- 6 numerical columns before encoding
- 8 object/categorical columns
- `salary` contains missing values

The notebook identifies **67 missing values in `salary`**, while the other columns contain no missing values. fileciteturn4file0

### Salary Statistics

The notebook calculates:

- Median salary: **265,000**
- Mean salary: **288,655.41**
- Mode salary: **300,000**

These statistics are calculated before the missing salary values are replaced. fileciteturn4file0

---

## 🧹 Data Preprocessing

### 1. Missing Value Handling

The notebook fills missing salary values with `0`:

```python
df['salary'].fillna(0, inplace=True)
```

After this operation, the notebook verifies that there are no remaining null values. fileciteturn4file0

This treatment effectively represents records without an available salary value as zero.

---

### 2. Categorical Encoding

Categorical variables are converted into numerical values using `LabelEncoder`.

The following columns are encoded:

```text
gender
ssc_b
hsc_b
hsc_s
degree_t
workex
specialisation
status
```

The notebook applies `LabelEncoder` separately to each column. fileciteturn6file0

---

### 3. Outlier Detection

Box plots are generated to inspect the numerical distributions and identify potential outliers. fileciteturn7file0

The notebook then uses **Feature-engine's `Winsorizer`** with:

```python
capping_method='iqr'
tail='both'
fold=1.5
```

Winsorization is applied to:

- `hsc_p`
- `degree_p`
- `salary`

Rather than deleting observations, the extreme values are capped according to the IQR-based approach. fileciteturn7file0

A second set of box plots is generated after this step to inspect the transformed distributions.

---

## 📏 Feature Scaling

The notebook uses `StandardScaler` to standardize all model-related columns, including the target salary column.

The scaling is applied to:

```text
gender
ssc_p
ssc_b
hsc_p
hsc_b
hsc_s
degree_p
degree_t
workex
etest_p
specialisation
mba_p
status
salary
```

Standardization transforms the values so that the resulting features have a mean close to 0 and standard deviation close to 1.

The target `salary` is also standardized before model training. fileciteturn5file0

---

## 🤖 Machine Learning Model

The notebook uses:

### Linear Regression

```python
model = LinearRegression()
model.fit(x_train, y_train)
```

The model uses 13 input features:

```text
gender
ssc_p
ssc_b
hsc_p
hsc_b
hsc_s
degree_p
degree_t
workex
etest_p
specialisation
mba_p
status
```

The target variable is:

```text
salary
```

The implementation is based on `sklearn.linear_model.LinearRegression`. fileciteturn5file0

---

## 🧪 Train/Test Split

The dataset is divided using:

```python
train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=42
)
```

This results in approximately:

- **70% training data — 150 records**
- **30% testing data — 65 records**

The fixed `random_state=42` makes the split reproducible. The notebook output shows 150 training observations. fileciteturn5file0

---

## 📈 Model Evaluation

The model is evaluated using three regression metrics.

### Mean Absolute Error (MAE)

Measures the average absolute difference between the actual and predicted values.

```text
MAE = 0.2823
```

### Root Mean Squared Error (RMSE)

Measures prediction error while giving greater weight to larger errors.

```text
RMSE = 0.4610
```

### R² Score

Measures the proportion of variance explained by the regression model.

```text
R² = 0.7863
```

These are the exact values produced by the saved notebook. fileciteturn5file0

### Important interpretation note

Because the notebook standardizes the `salary` column **before** splitting the data and training the model, the MAE and RMSE above are expressed in the **standardized salary scale**, not directly in currency units.

The reported R² is scale-independent, but the error metrics should not be interpreted as an average salary error of `0.28` currency units.

---

## 🔮 Prediction Example

The notebook also demonstrates prediction for a manually supplied standardized feature vector:

```python
model.predict([
    [0.739434, -0.028087, 1.082459, 2.335422,
     0.800763, -0.641955, -1.144306, 1.576284,
     -0.724446, -1.291091, 1.123903, -0.597647,
     0.672832]
])
```

The model returns:

```text
0.6949847
```

This prediction is also on the **standardized salary scale**, because the target was standardized during preprocessing. fileciteturn5file0

---

## 🛠️ Technologies Used

- **Python**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Feature-engine**
- **Jupyter Notebook**

---

## 📦 Installation

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn feature-engine
```

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd placement-salary-prediction
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn feature-engine
```

### 3. Add the dataset

The original dataset is **not included in this repository**.

Place the authorized `Placement_data.csv` file in your local environment and update the notebook's loading path:

```python
df = pd.read_csv("path/to/Placement_data.csv")
```

The current notebook contains a machine-specific Windows path, so it should be changed before sharing/running the notebook on another computer. fileciteturn4file0

### 4. Run the notebook

Open:

```text
placement_salary_prediction.ipynb
```

and execute the cells from top to bottom.

---

## ⚠️ Dataset Availability

The dataset used for this project was provided externally and is **not included in this public repository**.

This repository therefore contains the analysis and machine-learning notebook, but does not redistribute the original dataset.

---

## ⚠️ Important Technical Considerations

This project demonstrates a complete beginner/intermediate machine-learning workflow, but there are several areas that could be improved in a production-quality implementation.

### 1. Target leakage through missing-value treatment

The notebook replaces missing salary values with `0` and then uses salary as the regression target.

For a salary-prediction problem, missing salaries generally represent unavailable target labels rather than meaningful zero salaries. A stronger approach would normally train the salary model only on observations where salary is actually available.

### 2. Target scaling before train/test split

The notebook fits `StandardScaler` on the entire dataset before splitting into training and testing data.

A more rigorous approach would fit preprocessing transformations only on the training data to avoid data leakage.

### 3. Categorical encoding

`LabelEncoder` is used on predictor columns. For nominal categorical variables, one-hot encoding is often more appropriate because label encoding can introduce an artificial numerical ordering.

### 4. Placement status as a predictor

`status` is included as an input feature while predicting salary.

Because salary is naturally associated with placed candidates, this feature should be considered carefully when interpreting the model and defining the prediction scenario.

### 5. Small dataset

The model is trained on a relatively small dataset, with 215 original observations and 150 observations in the training split. Model performance may therefore vary with different train/test splits.

These points are presented as improvement opportunities rather than changes made to the original notebook.

---

## 📁 Recommended Repository Structure

Since the original dataset is not being publicly distributed, a clean repository can contain:

```text
placement-salary-prediction/
│
├── README.md
└── placement_salary_prediction.ipynb
```

You do **not** need to upload `Placement_data.csv` if you do not have permission to redistribute it.

---

## 🚀 Future Improvements

Possible improvements to this project include:

- Use a proper preprocessing pipeline with `ColumnTransformer`
- Use one-hot encoding for categorical variables
- Split the data before fitting preprocessing transformations
- Train the salary model only on records with valid salary targets
- Keep the target in its original salary units or inverse-transform predictions
- Compare Linear Regression with Ridge, Lasso, Random Forest, and other regression models
- Use cross-validation for more robust evaluation
- Add actual-vs-predicted plots
- Add residual analysis
- Analyze feature importance/model coefficients
- Save the trained model for future predictions
- Create a simple Streamlit interface for salary prediction

---

## 📌 Key Takeaway

This project demonstrates an end-to-end **placement salary prediction workflow** using Python and Linear Regression.

The workflow covers:

**Data Exploration → Missing Value Handling → Categorical Encoding → Outlier Treatment → Feature Scaling → Train/Test Split → Linear Regression → Prediction → Evaluation**

The final saved notebook reports an **R² score of approximately 0.7863** on its test set, with MAE and RMSE of approximately **0.2823** and **0.4610** respectively on the standardized salary scale. fileciteturn5file0

---

## 👩‍💻 Project

**Project:** Placement Salary Prediction  
**Model:** Linear Regression  
**Dataset:** Placement dataset provided externally  
**Task:** Regression / Salary Prediction
