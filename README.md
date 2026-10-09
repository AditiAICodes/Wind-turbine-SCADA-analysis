# Wind Turbine Performance Analysis & Anomaly Detection

**An exploratory machine learning project involving power prediction, feature engineering, performance deviation analysis, anomaly classification, and model interpretation.**

## 1. Project Overview

Wind turbines operate under changing environmental conditions, making it important to understand how wind conditions relate to power generation and how actual performance differs from expected output.

This project explores wind turbine operational data through an end-to-end experimental machine learning workflow. The work progresses from exploratory data analysis and data cleaning to feature engineering, regression modeling, anomaly identification, multiple classification experiments, hyperparameter optimization, and SHAP-based model interpretation.

The project investigates two main questions:

1. **Power prediction:** How accurately can turbine power output be estimated from the available wind-condition features?
2. **Anomaly detection:** Can deviations between expected and actual power generation help identify observations that may represent unusual operating behavior?

The notebook also investigates how feature engineering, anomaly thresholds, class-imbalance handling, and model configuration affect the results.

This is an exploratory learning project rather than a production-ready predictive maintenance system. Its purpose is to investigate the available data, experiment with different modeling approaches, and understand both the capabilities and limitations of machine learning for performance analysis.

## 2. Project Objectives

- Explore wind turbine operational data and its statistical characteristics.
- Analyze the relationship between wind speed and actual power generation.
- Compare actual power output with the theoretical power curve.
- Quantify deviations between theoretical, actual, and model-estimated power.
- Engineer features representing circular wind direction, nonlinear wind-speed relationships, and short-term temporal variation.
- Develop a Random Forest regression baseline for power prediction.
- Apply hyperparameter optimization to improve regression performance.
- Experiment with different power-deviation thresholds for candidate anomaly identification.
- Compare multiple Random Forest classification configurations.
- Investigate class imbalance using SMOTE and class-weighted learning.
- Evaluate models using precision, recall, F1-score, accuracy, MAE, and R².
- Use SHAP to investigate feature contributions to model predictions.
- Identify methodological limitations and potential directions for improvement.

## 3. Dataset Description

The project uses the `T1.csv` dataset, containing approximately 50,530 observations recorded at 10-minute intervals during 2018.

| Feature | Description |
|---|---|
| `Date` | Timestamp of each observation |
| `ActivePower` | Actual turbine power output, in kW |
| `WindSpeed` | Recorded wind speed, in m/s |
| `TheoreticalPower` | Theoretical power output, in kW |
| `WindDirection` | Wind direction, in degrees |

The dataset provides environmental measurements and power-generation information suitable for exploratory performance analysis.

However, it does not include verified fault labels or a comprehensive set of turbine-health measurements. This limits the conclusions that can be drawn about actual mechanical failures.

## 4. Methodology

### Stage 1: Data Inspection and Exploratory Data Analysis

The first stage focused on understanding the structure, quality, and statistical characteristics of the dataset.

Activities included:

- Inspecting the dataset dimensions, data types, and numerical summaries.
- Examining the ranges and distributions of active power, wind speed, theoretical power, and wind direction.
- Converting timestamps into datetime format.
- Investigating the relationship between wind speed and power generation.
- Visualizing actual power output against the theoretical power curve.
- Exploring differences between theoretical and observed performance.

The comparison between actual and theoretical power provided an initial way to investigate performance deviations before introducing machine learning models.

### Stage 2: Data Cleaning and Power-Loss Feature

Negative active-power values were clipped to zero during preprocessing.

A feature named `Power_Loss` was created to measure the difference between theoretical and actual power output.

```python
df['Power_Loss'] = (
    df['TheoreticalPower'] - df['ActivePower']
)
```

The calculation is:

**Power Loss = Theoretical Power − Actual Power**

Positive values indicate that actual generation is below theoretical output. Negative values indicate that actual generation exceeds the theoretical value.

This feature provided a straightforward measure of the performance gap. However, a positive gap does not necessarily indicate a turbine fault, because operating conditions and data quality can also influence the observed difference.

### Stage 3: Feature Engineering

Feature engineering was used to transform the original measurements into representations that could better capture their underlying relationships.

#### 3.1 Circular Encoding of Wind Direction

Wind direction is an angular variable. Directions of 359° and 1°, for example, are physically close even though their numerical values differ substantially.

To represent this circular relationship, wind direction was transformed into sine and cosine components:

```python
import numpy as np

df['Wind_Sin'] = np.sin(
    np.deg2rad(df['WindDirection'])
)

df['Wind_Cos'] = np.cos(
    np.deg2rad(df['WindDirection'])
)
```

This representation avoids treating the beginning and end of the angular scale as distant values.

#### 3.2 Nonlinear Wind-Speed Feature

A cubic wind-speed feature was introduced to explore a nonlinear representation of wind speed:

```python
df['Wind_Power_Cubed'] = df['WindSpeed'] ** 3
```

This transformation was motivated by the nonlinear relationship between wind speed and the potential power available in wind. It is an engineered representation, not an additional physical measurement.

#### 3.3 Rolling Time-Series Features

To explore short-term wind conditions, two rolling statistics were generated using a window of 18 observations, corresponding to three hours at 10-minute intervals:

```python
df['Wind_3hr_Avg'] = (
    df['WindSpeed'].rolling(window=18).mean()
)

df['Wind_3hr_Std'] = (
    df['WindSpeed'].rolling(window=18).std()
)
```

- `Wind_3hr_Avg` represents recent average wind speed.
- `Wind_3hr_Std` represents short-term variability in wind speed.

These features were intended to capture temporal context that individual observations might not represent.

The initial rolling window produces missing values, which were removed before the subsequent modeling experiment.

#### 3.4 Correlation Analysis

A correlation heatmap was generated to explore relationships among the numerical variables.

This helped investigate potential associations between wind conditions, actual power, theoretical power, and engineered features.

Correlation was used as an exploratory tool rather than evidence of causation.

### Stage 4: Rule-Based Candidate Anomaly Identification

An initial heuristic was applied to identify observations with relatively high wind speed but almost no active power output.

The rule was:

```python
df_healthy = df[
    ~(
        (df['WindSpeed'] > 3.5) &
        (df['ActivePower'] <= 10)
    )
]
```

This filtering step flagged 2,302 observations and retained 48,228 observations for the regression experiment.

The rule provided an initial assumption for constructing a working dataset. The flagged observations were not independently confirmed as equipment failures, and the suitability of the rule would require validation against turbine operating characteristics.

### Stage 5: Random Forest Regression

A Random Forest Regressor was trained to estimate actual power output from wind speed and wind direction.

**Input features:**

- `WindSpeed`
- `Wind_Sin`
- `Wind_Cos`

**Target variable:** `ActivePower`

The data was divided into training and test sets using an 80:20 random split with `random_state=42`.

The initial model used 100 decision trees.

**Initial regression results**

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 101.52 kW |
| R² Score | 96.27% |

The MAE measures the average absolute difference between predicted and observed power output. The R² score measures how much variation in the target was explained by the model on the evaluated test split.

These results established a baseline for subsequent optimization.

### Stage 6: Hyperparameter Optimization

Rather than retaining the initial regression model, RandomizedSearchCV was used to explore different Random Forest configurations.

The search considered the following hyperparameters:

| Hyperparameter | Values Explored |
|---|---|
| `n_estimators` | 100, 200, 300 |
| `max_depth` | 10, 15, 20, 25, `None` |
| `min_samples_split` | 2, 5, 10 |
| `min_samples_leaf` | 1, 2, 4 |

The experiment used 10 sampled parameter configurations and three-fold cross-validation.

**Best configuration identified**

| Hyperparameter | Selected Value |
|---|---:|
| `n_estimators` | 200 |
| `max_depth` | 10 |
| `min_samples_split` | 2 |
| `min_samples_leaf` | 4 |

**Regression performance comparison**

| Metric | Initial Model | Tuned Model |
|---|---:|---:|
| MAE | 101.52 kW | 94.83 kW |
| R² Score | 96.27% | 96.66% |

The tuned model reduced MAE by approximately 6.6% and improved the R² score by 0.39 percentage points.

This experiment demonstrated how hyperparameter search can improve predictive performance within the chosen evaluation setup.

### Stage 7: Expected Power and Deviation Analysis

The regression model was applied to generate model-estimated expected power for the available observations.

Two features were used to represent this comparison:

- `Expected_Power_kW`: Model-estimated expected power.
- `Power_Deviation_kW`: Difference between estimated expected power and actual power.

The deviation was calculated as:

```python
df['Power_Deviation_kW'] = (
    df['Expected_Power_kW'] - df['ActivePower']
)
```

A positive deviation means actual power is below the model's estimate. A negative deviation means actual power exceeds the estimate.

The maximum positive deviation observed was approximately 3,596.18 kW.

An initial candidate anomaly label was then generated using a threshold of 500 kW:

```python
df['Anomaly_Alarm'] = (
    df['Power_Deviation_kW'] > 500
).astype(int)
```

This produced 2,043 observations labeled as candidate anomalies.

These labels represented large positive deviations from the regression model's estimate. They were used as experimental targets for the subsequent classification models, not as confirmed fault records.

### Stage 8: Classification Experiment A: 500 kW Threshold and SMOTE

The first classification experiment used the candidate anomaly labels generated from the 500 kW threshold.

The input features were:

- `WindSpeed`
- `Wind_Sin`
- `Wind_Cos`

A Random Forest Classifier with 100 trees was trained to predict the binary anomaly labels.

Because the anomaly class was smaller than the non-anomaly class, SMOTE was applied to the training data. The test data remained unchanged.

After resampling, the training set contained 38,828 observations in each class.

**Classification results**

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Non-anomaly (0) | 0.98 | 0.85 | 0.91 |
| Anomaly (1) | 0.15 | 0.55 | 0.23 |

**Overall accuracy: 84%**

The model achieved an anomaly recall of 0.55, but anomaly precision was only 0.15. This indicates that many observations predicted as anomalies did not belong to the positive class defined by the generated labels.

The experiment highlighted that balancing the training data does not automatically produce strong minority-class performance.

### Stage 9: Classification Experiment B: Stricter 1,000 kW Threshold

A stricter anomaly threshold was introduced to focus on larger deviations:

```python
df['Strict_Anomaly_Alarm'] = (
    df['Power_Deviation_kW'] > 1000
).astype(int)
```

The classification configuration was changed to:

- 200 decision trees.
- Maximum tree depth of 15.
- `class_weight='balanced_subsample'`.

This experiment used class weighting instead of the SMOTE-based approach used in Experiment A.

**Classification results**

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Non-anomaly (0) | 0.98 | 0.93 | 0.96 |
| Anomaly (1) | 0.16 | 0.48 | 0.24 |

**Overall accuracy: 92%**

Overall accuracy increased to approximately 92%, but anomaly precision remained low and recall fell to 0.48.

The threshold changed the definition and prevalence of the positive class, so this result should not be interpreted as a directly equivalent comparison with Experiment A.

The experiment demonstrated why classification accuracy must be considered alongside minority-class metrics.

### Stage 10: Classification Experiment C: Engineered Temporal and Nonlinear Features

A further classification experiment incorporated the engineered wind-speed and temporal features.

**Input features:**

- `WindSpeed`
- `Wind_Sin`
- `Wind_Cos`
- `Wind_3hr_Avg`
- `Wind_3hr_Std`
- `Wind_Power_Cubed`

**Target:** `Strict_Anomaly_Alarm`

The model used 200 trees, a maximum depth of 15, and `class_weight='balanced_subsample'`.

**Final classification results**

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Non-anomaly (0) | 0.99 | 0.95 | 0.97 |
| Anomaly (1) | 0.24 | 0.57 | 0.34 |

**Overall accuracy: 94%**

Compared with Experiment B, the final classifier improved the anomaly-class metrics:

| Anomaly Metric | Experiment B | Experiment C |
|---|---:|---:|
| Precision | 0.16 | 0.24 |
| Recall | 0.48 | 0.57 |
| F1-Score | 0.24 | 0.34 |

The engineered features may have helped the model distinguish the generated anomaly labels. However, these results do not establish the ability to detect actual turbine failures because the target labels were derived from power-deviation thresholds.

### Stage 11: SHAP-Based Model Interpretation

SHAP was used to investigate how input features contributed to the final Random Forest classifier's predictions.

A TreeExplainer was created for the trained classifier, and a sample of 1,000 observations from the test set was selected for analysis.

A SHAP summary plot was generated for the anomaly class.

This step extended the workflow beyond predictive metrics by exploring how the model used the engineered features when making predictions.

SHAP values explain model behavior for the evaluated observations. They do not establish physical causation or identify the mechanical cause of a power deviation.

## 5. Results Summary

| Experiment | Main Result | Interpretation |
|---|---|---|
| Initial Random Forest regression | MAE: 101.52 kW | Established a power-prediction baseline |
| Tuned Random Forest regression | MAE: 94.83 kW | Reduced prediction error |
| 500 kW anomaly threshold with SMOTE | Accuracy: 84% | Anomaly precision remained low |
| 1,000 kW threshold classifier | Accuracy: 92% | Higher accuracy, but limited anomaly precision |
| Classifier with engineered features | Accuracy: 94% | Improved anomaly precision, recall, and F1-score |
| SHAP interpretation | Summary plot generated | Explored feature contributions to predictions |

**Important distinction:** The regression R² score of 96.66% measures how well the regression model explained variation in active power on its test split. The final classifier's 94% accuracy measures agreement with the generated anomaly labels. Neither figure represents verified fault-detection accuracy.

## 6. Key Findings and Learnings

### Feature engineering changes the representation of a problem

Circular encoding, nonlinear transformations, and rolling statistics provided different ways to represent the original measurements and investigate their relationships with power generation and generated anomaly labels.

### Hyperparameter optimization can improve a baseline

The regression experiment showed a reduction in MAE from 101.52 kW to 94.83 kW after hyperparameter tuning.

### High accuracy does not guarantee strong anomaly detection

The classification experiments achieved overall accuracy as high as 94%, while final anomaly precision remained 0.24.

This means that the model's positive predictions still included a substantial proportion of observations that were not positive according to the generated labels.

### Class balancing is not a complete solution

SMOTE and class weighting were explored to address class imbalance. These techniques changed the training process, but did not eliminate the difficulty of identifying the minority class.

### Generated labels determine what the classifier learns

Because anomaly labels were generated from model-based power deviations, the classifier learned patterns associated with those labels. This is different from learning directly from independently confirmed equipment failures.

### Model performance depends on the available data

The dataset supports exploratory power-performance analysis, but it cannot independently establish the mechanical causes of observed deviations.

## 7. Limitations

### 7.1 Limited Turbine-Health Measurements

The dataset includes wind speed, wind direction, actual power, theoretical power, and timestamps. These variables provide useful context for performance analysis, but do not fully describe the physical condition of the turbine.

Measurements such as vibration, temperature, rotor speed, component status, maintenance history, and verified failure records could provide additional evidence for distinguishing genuine equipment problems from normal operating variation.

### 7.2 No Independently Verified Fault Labels

The anomaly targets were constructed using heuristic filtering and thresholds applied to power deviations.

Consequently, the classification experiments evaluate agreement with generated labels rather than independently verified mechanical failures.

### 7.3 Potential Temporal Evaluation Bias

Random train-test splits were used in the notebook. Because neighboring time-series observations can be similar, random splitting may produce optimistic estimates of generalization to future observations.

Chronological evaluation would provide a more realistic assessment of future performance.

The rolling features should also be generated after explicitly sorting the data by timestamp and ensuring that each feature uses only information available at the prediction time.

### 7.4 Class Imbalance and False Positives

The anomaly class was relatively small, so overall accuracy did not fully represent anomaly-class performance.

Future evaluation should emphasize precision, recall, F1-score, confusion matrices, and precision-recall curves. The acceptable trade-off between false alarms and missed events would depend on the intended application.

### 7.5 Exploratory Preprocessing and Thresholds

Clipping negative power values and filtering observations using wind-speed conditions were exploratory preprocessing decisions.

Similarly, the 500 kW and 1,000 kW thresholds were experimental choices rather than validated universal fault criteria. These decisions would need to be tested against the data-collection process and turbine operating characteristics.

### 7.6 Generalization Across Turbines and Time Periods

The experiments were conducted on one dataset. Their results do not establish that the models would perform similarly on other turbines, operating environments, or time periods.

### 7.7 Interpretation Does Not Establish Causality

SHAP can help explain how input features contribute to model predictions, but feature contributions alone cannot identify the physical cause of a power deviation.

## 8. Future Improvements

Potential next steps include:

- Introduce chronological train-test splits and time-aware cross-validation.
- Validate timestamp ordering and preprocessing assumptions.
- Compare alternative regression and anomaly-detection algorithms.
- Investigate precision-recall trade-offs when selecting anomaly thresholds.
- Explore approaches that do not rely exclusively on model-generated binary labels.
- Incorporate additional turbine-condition measurements and verified fault records.
- Evaluate performance on separate time periods or additional turbines.
- Improve SHAP visualizations and compare feature contributions across model configurations.
- Validate findings with relevant turbine-operating knowledge before making claims about actual faults.

## 9. Technology Stack

- **Programming:** Python
- **Data manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** Scikit-learn
- **Class-imbalance handling:** Imbalanced-learn (SMOTE)
- **Model interpretation:** SHAP
- **Development environment:** Jupyter Notebook / Google Colab

## 10. Project Status

**Completed exploratory learning project.**

This project documents an experimental workflow covering exploratory data analysis, feature engineering, regression, hyperparameter optimization, threshold-based anomaly labeling, multiple classification configurations, class-imbalance handling, and model interpretation.

The results are preliminary and should not be treated as validated industrial findings.

## Conclusion

This project explores how machine learning can be used to analyze wind turbine power generation, estimate expected output, investigate performance deviations, and classify observations using generated anomaly labels.

The work demonstrates an iterative approach to modeling: establishing a baseline, engineering additional features, comparing model configurations, evaluating classification metrics, and interpreting predictions.

An equally important outcome is understanding the limits of the results. Strong regression performance and high classification accuracy do not automatically establish reliable real-world fault detection. The next step toward a more robust system would be to strengthen temporal evaluation, validate anomaly labels, and incorporate additional operational evidence.

**Disclaimer:** This project is an educational and exploratory prototype. It is not a validated industrial fault-detection or predictive-maintenance system.
