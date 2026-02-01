# NFL Game Win Predictor
This is an end-to-end machine learning application that predicts the probability of a **home team winning an NFL game** using historical team performance metrics and recent form.

This project covers the **full ML lifecycle**, which includes data collection, feature engineering, EDA, modeling, evaluation, and deployment as a live web app.

**Live App**: https://nfl-win-predictor.streamlit.app/

## Project Motivation
Predicting sports outcomes is a classic yet challenging real-world machine learning problem due to temporal dependencies, data leakage risks, and the need for
**interpretable probabilities** rather than simply classifications. This project was designed to simulate how sports analytics models are built in industry, 
emphasise proper feature engineering and evaluation, and deliver a decision-supported tool, not just a model.

## Problem Statement
Given an NFL matchup, predict the probability that the home team wins, given only the information available before the game is played.

The target variable is `home_win` (1 = home team wins, 0 = otherwise)

## Data
The data was collected from the **ESPN NFL API**. I collected:
* Game schedules and results
* Team-level statistics
Regular season and post-season data was collected from **2014-2025** seasons.

Data was collected programmatically and processed in a reproducible pipeline.

## Feature Engineering
### Rolling Team Performance Metrics
To capture recent form while avoiding data leakage, rolling averages were computed per team using only past games:
* Points scored/allowed (last 3 & 5 games)
* Yards per play
* First downs
* Time of possession
* Red zone attempts
* Turnovers

### Home vs. Away Feature Differencing
For each game, features were transformed into home − away differences, which:
* reduces multicollinearity,
* improves model stability,
* and mirrors how betting and analytics systems reason about matchups.
For example:
```
diff_points_for_avg_last_5 = home_points_for_avg_last_5 - away_points_for_avg_last_5
```
## Modeling Approach
### Baseline Models
* Logistic Regression
* Random Forest
* Gradient Boosting
* XGBoost

### Evaluation Strategy
* Stratified K-Fold Cross Validation
* Primary metric: **ROC-AUC**
* Secondary metrics:
    * Log loss (probability quality)
    * Accuracy (interpretability quality)

### Model Performance
| Model | ROC-AUC | Log loss | Accuracy |
| ---------- | ---------- | ---------- | ---------- |
| Logistic Regression | 0.6727 | 0.64455 | 0.6285 |
| Random Forest | 0.6628 | 0.6504 | 0.6239 |
| Gradient Boosting | 0.6347 | 0.6750 | 0.6009 |
| XGBoost | 0.6425 | 0.6712 | 0.6032 | 

**Logistic Regression** was selected due to:
* strongest ROC-AUC
* interpretability

### Hyperparameter Tuning
Logistic Regression was tuned using Grid Search. T
```
C = 0.1
Penalty = L1
Solver = liblinear
```
We achieved an ROC-AUC score of **0.6751** using the tuned hyperparameters, and a test ROC-AUC score of 0.6709.

The final pre‑game win‑prediction model achieved an out‑of‑sample accuracy of 63.31% when evaluated on the held‑out test seasons. 
This level of performance is meaningfully higher than simple baselines such as always picking the home team or the closing favorite, 
which typically fall in the mid‑50s to at most around 60% accuracy over long samples. It is also in the range reported by many public 
and academic models that aim to approximate betting‑market quality, though it should still be interpreted as an approximation rather 
than a replacement for full market odds. Overall, a 63.31% hit rate places this model in a competitive tier for pre‑game, stats‑only prediction, 
especially given its relatively simple feature set and modeling approach.

## Deployment
The final model was deployed using **Streamlit Community Cloud**.
#### App Features
* User selects Home Team and Away Team
* App automatically:
*   retrieves latest team stats
*   computes feature differences
*   applies scaling
*   generates a win probability
* Displays:
*   predicted home win probability
*   feature-level impact for transparency

## Tech Stack
* Python
* pandas, numpy
* scikit-learn
* joblib
* Streamlit
* ESPN API

## Future Improvements
* Automated weekly data refresh
* Injury and roster availability signals
* Incorporating weather data
* Real-time win probability during games

## Author
**Janhavi Tamhankar**

MS Data Science and Statistics, UT Dallas

[LinkedIn](https://www.linkedin.com/in/janhavitamhankar/)

