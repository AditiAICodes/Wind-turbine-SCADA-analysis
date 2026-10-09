# Wind Turbine Performance Analysis & Anomaly Detection

### An Exploratory Machine Learning Project for Power Prediction, Performance Deviation Analysis, and Anomaly Classification

## 1. Project Overview

Wind turbines operate under changing environmental conditions, making it important to understand how wind conditions relate to power generation and how actual performance differs from expected output.

This project explores wind turbine operational data through a complete experimental machine learning workflow, progressing from data exploration and feature engineering to regression modeling, rule-based anomaly identification, classification experiments, hyperparameter optimization, and model interpretation.

The project investigates two connected questions:

1. **Power prediction:** How accurately can turbine power output be estimated from the available wind-condition features?
2. **Anomaly detection:** Can deviations between expected and actual power generation help identify observations that may represent unusual operating behavior?

Rather than treating a high model score as the final objective, this project also examines the limitations of the available data, the effect of class imbalance, and the distinction between identifying unusual observations and confirming actual turbine faults.

**Project status:** Completed exploratory prototype. Results are preliminary and have not been validated against confirmed equipment failures.

## 2. Project Objectives

- Investigate the relationship between wind speed and turbine power generation.
- Compare actual power output with the theoretical power curve.
- Quantify deviations between theoretical, actual, and model-estimated power.
- Engineer features that represent circular wind direction, nonlinear wind-speed relationships, and short-term temporal variation.
- Develop a regression baseline for estimating expected power output.
- Experiment with multiple anomaly-labeling thresholds and Random Forest classification configurations.
- Explore class-imbalance handling using SMOTE and class-weighted learning.
- Evaluate models using regression and classification metrics.
- Apply SHAP to investigate feature contributions to anomaly predictions.
- Identify limitations in the data and modeling approach and outline possible improvements.

## 3. Dataset Description

The project uses the wind turbine dataset `T1.csv`, containing approximately 50,530 observations recorded at 10-minute intervals during 2018.

| Feature | Description |
|---|---|
| Date | Timestamp of the observation |
| ActivePower | Actual turbine power output, in kW |
| WindSpeed | Recorded wind speed, in m/s |
| TheoreticalPower | Theoretical power output, in kW |
| WindDirection | Wind direction, in degrees |

The dataset provides environmental measurements and power-generation information that can be used to explore turbine performance.

However, it does not contain verified fault labels or a comprehensive set of turbine-health measurements. This distinction becomes important when interpreting the anomaly-detection experiments.

## 4. End-to-End Workflow

### Stage 1: Data Inspection and Exploratory Data Analysis

The initial analysis focused on understanding the structure, quality, and statistical characteristics of the dataset.

Activities included:

- Inspecting column names, data types, missing values, and dataset dimensions.
- Examining descriptive statistics, distributions, and ranges of the numerical variables.
- Converting the timestamp column into a proper datetime representation.
- Investigating active power, wind speed, theoretical power, and wind direction.
- Visualizing actual power generation against wind speed.
- Comparing actual generation with the theoretical power curve to investigate differences between expected and observed behavior.

The comparison between actual and theoretical power provided an initial way to explore performance deviations before introducing machine learning models.

### Stage 2: Data Cleaning and Power-Deviation Analysis

Negative active-power observations were clipped to zero for the initial analysis.

A feature named `Power_Loss` was created:

\[
\text{Power\_Loss}
=
\text{TheoreticalPower}
-
\text{ActivePower}
\]

Positive values represent observations where actual generation is below theoretical output.

This feature provided a direct way to quantify the gap between theoretical and observed performance. However, the gap alone does not establish that a fault occurred, since operational conditions and data quality can also affect the observed difference.

### Stage 3: Feature Engineering

Feature engineering was used to transform the available variables into representations that could better capture their underlying relationships.

**Circular encoding of wind direction**

Wind direction is an angular variable, meaning values close to 0° and 360° represent similar directions. Treating these values as ordinary numbers can introduce an artificial distance between them.

To address this, two features were created:

\[
\text{Wind\_Sin}=\sin(\theta)
\]

\[
\text{Wind\_Cos}=\cos(\theta)
\]

Here, \(\theta\) is the wind direction converted from degrees to radians.

This representation preserves the circular nature of the direction measurement.

**Nonlinear wind-speed representation**

A cubic wind-speed feature was introduced:

\[
\text{Wind\_Power\_Cubed}
=
\text{WindSpeed}^{3}
\]

This transformation was used to investigate whether a nonlinear representation of wind speed could provide useful information for modeling power-related behavior.

**Rolling time-series features**

Using a window of 18 observations, corresponding to three hours at 10-minute intervals, two additional features were generated:

- `Wind_3hr_Avg`: Rolling average of wind speed.
- `Wind_3hr_Std`: Rolling standard deviation of wind speed.

These features were intended to represent recent wind conditions and short-term variability rather than relying exclusively on individual measurements.

The initial rolling window produces missing values, which were removed before the subsequent modeling experiment.

**Correlation analysis**

A correlation heatmap was also generated to examine relationships among numerical variables and help explore potential feature relationships.

### Stage 4: Rule-Based Candidate Anomaly Identification

Before training the classifiers, a simple heuristic was used to remove observations with relatively high wind speed but almost no active power:

`WindSpeed > 3.5` and `ActivePower <= 10`

This rule flagged 2,302 observations, leaving 48,228 observations in the filtered dataset.

The rule was used to construct an initial working dataset for regression modeling.

These observations were not independently confirmed as faulty turbine events. The filtering rule represents an exploratory assumption, and its suitability would require further validation using turbine-specific operating knowledge.

### Stage 5: Random Forest Regression

A Random Forest Regressor was trained to estimate active power using the following inputs:

- `WindSpeed`
- `Wind_Sin`
- `Wind_Cos`

The target variable was `ActivePower`.

The data was split into training and test sets using an 80:20 random split with `random_state=42`.

The initial model used 100 decision trees. Its performance was evaluated using Mean Absolute Error (MAE) and the coefficient of determination (R²).

**Initial regression results**

| Metric | Result |
|---|---:|
| Mean Absolute Error | 101.52 kW |
| R² Score | 96.27% |

The results indicated that the model could estimate active power reasonably well on the randomly held-out observations from the filtered dataset.

### Stage 6: Hyperparameter Optimization

Instead of retaining the initial configuration, RandomizedSearchCV was used to explore different Random Forest hyperparameter combinations.

The search considered:

- Number of trees: 100, 200, or 300.
- Maximum tree depth: 10, 15, 20, 25, or unlimited.
- Minimum samples required to split a node: 2, 5, or 10.
- Minimum samples required at a leaf: 1, 2, or 4.

The experiment used 10 sampled parameter configurations and three-fold cross-validation.

**Best configuration identified**

| Hyperparameter | Selected Value |
|---|---:|
| `n_estimators` | 200 |
| `max_depth` | 10 |
| `min_samples_split` | 2 |
| `min_samples_leaf` | 4 |

**Optimized regression results**

| Metric | Initial Model | Tuned Model |
|---|---:|---:|
| MAE | 101.52 kW | 94.83 kW |
| R² Score | 96.27% | 96.66% |

The tuned model reduced MAE by approximately 6.6% relative to the initial model and improved the R² score by 0.39 percentage points.

This experiment demonstrated how hyperparameter search can improve model performance, although the evaluation setup would need to be strengthened before drawing conclusions about performance on future time periods.

### Stage 7: Model-Based Expected Power and Deviation Analysis

The regression model was then applied to the available observations to generate model-estimated expected power.

Two quantities were distinguished:

- `Expected_Power_kW`: Power estimated by the regression model.
- `Power_Deviation_kW`: Difference between estimated expected power and actual power.

The deviation was defined as:

\[
\text{Power\_Deviation\_kW}
=
\text{Expected\_Power\_kW}
-
\text{ActivePower}
\]

A positive deviation indicates that actual power is below the model's estimate. A negative deviation indicates that actual power exceeds the estimate.

The maximum positive deviation observed was approximately 3,596.18 kW.

An initial anomaly flag was created using a positive deviation threshold of 500 kW:

\[
\text{Anomaly\_Alarm}
=
\begin{cases}
1, & \text{Power deviation}>500\text{ kW}\\
0, & \text{otherwise}
\end{cases}
\]

This produced 2,043 flagged observations.

The threshold-based labels served as a starting point for classification experiments. They represent deviations from a model-based baseline, not independently verified turbine failures.

### Stage 8: Random Forest Classification Experiments

After generating anomaly labels, multiple classification configurations were explored to investigate how threshold selection and class-imbalance handling affected the results.

#### Experiment A: Classification with a 500 kW Threshold

A Random Forest Classifier with 100 trees was trained using:

- `WindSpeed`
- `Wind_Sin`
- `Wind_Cos`

The target was the binary anomaly label generated from the 500 kW deviation threshold.

Because the anomaly class was much smaller than the non-anomaly class, SMOTE was applied to the training data to balance the classes. The test data was not resampled.

The resulting training set contained 38,828 observations in each class.

**Classification results**

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Non-anomaly (0) | 0.98 | 0.85 | 0.91 |
| Anomaly (1) | 0.15 | 0.55 | 0.23 |

Overall accuracy was approximately 84%.

Although anomaly recall reached 0.55, anomaly precision was only 0.15. This means many observations predicted as anomalies were not labeled as anomalies in the test set.

The experiment highlighted that balancing the training data does not automatically produce a reliable classifier for the minority class.

#### Experiment B: Classification with a Stricter 1,000 kW Threshold

A stricter threshold was then introduced:

\[
\text{Strict\_Anomaly\_Alarm}
=
\begin{cases}
1, & \text{Power deviation}>1000\text{ kW}\\
0, & \text{otherwise}
\end{cases}
\]

The classifier configuration was changed to:

- 200 decision trees.
- Maximum depth of 15.
- `class_weight='balanced_subsample'`.

This approach used class weighting rather than the SMOTE-based balancing experiment.

**Classification results**

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Non-anomaly (0) | 0.98 | 0.93 | 0.96 |
| Anomaly (1) | 0.16 | 0.48 | 0.24 |

Overall accuracy increased to approximately 92%.

However, anomaly precision remained low at 0.16, while recall fell to 0.48. The stricter threshold reduced the number of positive labels and changed the classification task, so its results are not a directly equivalent comparison with Experiment A.

The experiment reinforced the importance of evaluating the minority class separately from overall accuracy.

#### Experiment C: Classification with Engineered Temporal and Nonlinear Features

A further Random Forest classification experiment incorporated the additional engineered features:

- `WindSpeed`
- `Wind_Sin`
- `Wind_Cos`
- `Wind_3hr_Avg`
- `Wind_3hr_Std`
- `Wind_Power_Cubed`

The target remained the stricter 1,000 kW anomaly label.

The classifier used 200 trees, a maximum depth of 15, and `class_weight='balanced_subsample'`.

**Final classifier results**

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Non-anomaly (0) | 0.99 | 0.95 | 0.97 |
| Anomaly (1) | 0.24 | 0.57 | 0.34 |

Overall accuracy reached approximately 94%.

Compared with Experiment B, the anomaly-class precision increased from 0.16 to 0.24, recall increased from 0.48 to 0.57, and the F1-score increased from 0.24 to 0.34.

These results suggest that the additional features may have helped the classifier distinguish the generated labels. However, the experiment does not prove that the classifier can detect real equipment failures, because the target labels themselves were generated from model-based thresholds.

### Stage 9: SHAP-Based Model Interpretation

SHAP was used to investigate feature contributions to the final Random Forest classifier's predictions.

A TreeExplainer was created for the trained classifier, and a sample of 1,000 observations from the test set was selected for interpretation.

A SHAP summary plot was generated for the anomaly class.

This step extended the project beyond predictive performance metrics by exploring how the input features contributed to the model's anomaly predictions.

SHAP values explain the behavior of the trained model for the selected data. They do not establish physical causation or prove that a feature is responsible for a real turbine fault.

## 5. Results at a Glance

| Experiment | Main Result | Key Observation |
|---|---|---|
| Initial Random Forest regression | MAE: 101.52 kW | Established a power-prediction baseline |
| Tuned Random Forest regression | MAE: 94.83 kW | Improved regression performance |
| 500 kW anomaly threshold | Accuracy: 84% | Low anomaly precision despite SMOTE |
| 1,000 kW threshold classifier | Accuracy: 92% | Higher accuracy, but low anomaly precision |
| Classifier with engineered features | Accuracy: 94% | Improved anomaly precision, recall, and F1-score |
| SHAP interpretation | Summary plot generated | Explored feature contributions to predictions |

**Important distinction:** The regression R² score of 96.66% measures how well the model explained variation in active power on the test split. The final classifier's 94% accuracy measures agreement with generated anomaly labels. Neither metric represents 94% or 96.66% verified fault-detection accuracy.

## 6. Key Findings and Learnings

This project provided several practical lessons about working with real-world-style operational data.

**Feature engineering changes the representation of a problem.** Circular encoding, nonlinear transformations, and rolling statistics allowed the modeling experiments to represent wind direction, wind-speed relationships, and short-term variation in different ways.

**Model selection is an iterative process.** The initial regression model was followed by hyperparameter tuning, while the classification experiments explored different thresholds, class-balancing strategies, and feature sets.

**Accuracy can hide weak minority-class performance.** The classifiers achieved high overall accuracy while anomaly precision remained comparatively low. For a rare-event problem, false alarms and missed anomalies must be evaluated explicitly.

**Labels determine what a classifier learns.** When anomaly labels are created from a heuristic or a model-based threshold, the classifier learns to reproduce patterns associated with those labels. This is different from learning directly from confirmed equipment failures.

**Data availability determines the scope of the conclusions.** The dataset supports exploratory analysis of power output and wind conditions, but it does not provide enough independent evidence to establish the mechanical causes of observed deviations.

## 7. Limitations and Areas for Improvement

The limitations below are not just implementation details. They affect what the project can legitimately conclude.

### 7.1 Limited Operational and Condition-Monitoring Data

The dataset contains timestamps, wind speed, wind direction, actual power, and theoretical power. These variables are useful for performance analysis, but they provide only a partial view of the turbine's condition.

Additional information such as vibration, temperature, rotor speed, component status, maintenance history, and confirmed fault records could help distinguish genuine equipment problems from normal operating variation.

### 7.2 Generated Labels Rather Than Confirmed Failures

The anomaly labels were created using power-deviation thresholds, not independent fault records.

Consequently, the classifier experiments evaluate how well a model can reproduce the generated labeling scheme. They do not independently validate the detection of mechanical failures.

A stronger evaluation would use verified fault or maintenance events, where available, or clearly defined anomaly criteria supported by domain expertise.

### 7.3 Potential Temporal Leakage and Evaluation Bias

The notebook uses random train-test splits for the modeling experiments. With time-series data, nearby observations can be highly similar, meaning a random split may produce optimistic estimates of future performance.

A chronological split would better test whether a model trained on earlier observations generalizes to later observations.

Rolling features should also be calculated after explicitly sorting observations by timestamp. Their construction should respect the prediction time and avoid using information that would not have been available at that point.

### 7.4 Class Imbalance and False Alarms

The anomaly class represents a small proportion of the test observations. This can make overall accuracy look strong even when a considerable proportion of predicted anomalies are false positives.

Future evaluation should include precision-recall curves, confusion matrices, F1-score, and, where appropriate, precision-recall AUC. The relative cost of false alarms and missed failures should guide threshold selection.

### 7.5 Data Cleaning and Threshold Assumptions

Clipping negative active-power values and removing observations using a wind-speed rule were exploratory preprocessing decisions. Such choices may be reasonable starting points, but they require validation against the data-collection process and turbine operating characteristics.

Likewise, thresholds of 500 kW and 1,000 kW were experimental choices rather than universally valid fault criteria.

### 7.6 Model Complexity and Generalization

Random Forest models can learn useful nonlinear patterns, but performance on one dataset does not establish generalization to other turbines, seasons, or operating environments.

Further experiments could compare alternative regression and anomaly-detection approaches, assess simpler baselines, and validate results on a separate time period or turbine.

### 7.7 Interpretability Is Not Causality

SHAP helps explain how a model uses its input features. However, feature importance or contribution does not identify the physical cause of a power deviation. Establishing causality would require additional evidence and domain knowledge.

## 8. Future Work

Possible next steps include:

- Introduce chronological model evaluation and time-aware validation.
- Verify timestamp ordering and preprocessing assumptions.
- Compare additional regression models and anomaly-detection techniques.
- Evaluate anomaly thresholds using precision-recall trade-offs.
- Investigate alternative approaches that do not depend exclusively on generated binary labels.
- Incorporate additional turbine-condition measurements and verified maintenance or fault records.
- Evaluate generalization across different time periods or turbines.
- Improve SHAP visualizations and compare feature contributions across model configurations.

## 9. Technology Stack

- **Programming:** Python
- **Data manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** Scikit-learn
- **Class-imbalance handling:** Imbalanced-learn (SMOTE)
- **Model interpretation:** SHAP
- **Development environment:** Jupyter Notebook / Google Colab

## 10. Conclusion

This project demonstrates an exploratory end-to-end workflow for analyzing wind turbine performance data, engineering features, estimating power output, constructing candidate anomaly labels, experimenting with classification strategies, and interpreting model predictions.

Its main value lies in both the modeling experiments and the lessons learned from their limitations. The results show that regression performance, classification accuracy, and genuine fault-detection capability are different things.

The next step toward a more reliable anomaly-detection system would be to strengthen temporal evaluation, validate anomaly labels, and incorporate additional operational evidence.
