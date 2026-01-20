import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer

"""##1. Data Loading (5 Marks)"""


df = pd.read_csv('insurance.csv')
print('shape',df.shape)
print('columns : ',df.columns)
print(df.head(10))

"""##2. Data Preprocessing (10 Marks)"""

# this df has not missing values so handling missing values is required.

#encoding
df2 = df.copy()
cat_cols = ['sex','smoker','region']

oe = OrdinalEncoder(dtype=int)
df2[cat_cols] = oe.fit_transform(df2[cat_cols])

# outlier detection
def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers.index.tolist()
outliers_cols = 'bmi'
outliers = detect_outliers_iqr(df2[outliers_cols])
print(f"Found {len(outliers)} outliers in {outliers_cols}")

# scaling
numeric_features = ['age','bmi']
sc = StandardScaler()
sc_data = sc.fit_transform(df2[numeric_features])
df_scaled = pd.DataFrame(sc_data,columns=numeric_features)
print('=====scaled======')
print(df_scaled.head())

#correlation
print('=====correlation=====')
num_cols = ["age",	"bmi",	"children"]
target_cols = 'charges'

corr_matrix = df2[num_cols + [target_cols]].corr()
print(corr_matrix)

"""## Pipeline Creation, Primary Model Selection and Model Training

**Model** : Randon Forest


**Why** : Medical costs do not follow straight line. Insurance charges are driven by complex "if-then" scenarios that Random Forest is naturally designed to handle.
"""

# Pipeline Creation
num_cols = ["age", "bmi", "children"]
cat_cols = ['sex', 'smoker','region']
x = df[num_cols + cat_cols]
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Create preprocessing ONLY (ColumnTransformer)
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# Primary Model Selection
ml_pipeline = Pipeline([
    ('preprocessor', preprocessor),
     ('model', RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# Model Training
# Now fit once
ml_pipeline.fit(X_train, y_train)

# Predict
y_pred = ml_pipeline.predict(X_test)

# Score
accuracy = ml_pipeline.score(X_test, y_test)
print(f"Pipeline R² Score: {accuracy:.4f}")

"""## Cross-validation"""

# 3. Cross-validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(
    ml_pipeline,
    X_train,
    y_train,
    cv=5,
    scoring='r2'
)

print(f"CV Scores: {cv_scores}")
print(f"Average CV R²: {cv_scores.mean():.4f}")
print(f"STD CV R²: {cv_scores.std():.4f}")

"""## Hyperparameter Tuning and Best Model Selection"""

from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'model__n_estimators': [50, 100],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2', None]
}

# Create GridSearchCV
grid_search = GridSearchCV(
    estimator=ml_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    return_train_score=True
)

# Fit GridSearch
print("Starting Grid Search...")
grid_search.fit(X_train, y_train)

# Display results
print("\n" + "="*60)
print("GRID SEARCH RESULTS")
print("="*60)
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score (R²): {grid_search.best_score_:.4f}")
print(f"Number of combinations tested: {len(grid_search.cv_results_['params'])}")

# Get best model
best_model = grid_search.best_estimator_

"""##  Model Performance Evaluation"""

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
print(f"Best Model R² Score: {best_model.score(X_test, y_test):.4f}")
print(f"Best Model MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"Best Model MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"Best Model RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"Best Model R2: {r2_score(y_test, y_pred):.4f}")