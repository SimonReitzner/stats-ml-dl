# Imports
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Import data (already encoded)
df = pd.read_csv("../kidney_transplant.csv", header=0)
y = Surv().from_dataframe("death", "time", df)
X = df.drop(columns=["death", "time"])

# Convert to type float, because of FastSurvivalSVM
X = X.astype(float)

# Split data into train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=0
)

# Create grid of parameters to tune
param_grid = [
    {
        "alpha": 2. ** np.arange(-10, 11, 2),
        "max_iter": [50, 100, 1000]
    },
 ]

 # Definition of a simple scoring function for grid search
def scoring_cindex(model, X, y):
    """Return concordance_index_censored"""
    prediction = model.predict(X)
    result = concordance_index_censored(y["death"], y["time"], prediction)
    return result[0]

# Instantiate a random survival forest, a shufflesplit and a grid search
model = FastSurvivalSVM(fit_intercept=False, rank_ratio=1.0, tol=1e-5, random_state=0)
cv = ShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
gscv = GridSearchCV(model, param_grid,
    scoring=scoring_cindex,
    n_jobs=3,
    refit=False,
    cv=cv,
    verbose=2
)

# Tune the model
gscv.fit(X_train, y_train)

# Set best parameters (from tuning) and fit final model
model.set_params(**gscv.best_params_)
model.fit(X_train, y_train)

# Get the test score
print(f"Test score of the tuned model: {model.score(X_test, y_test)}")

# Build a subset of the test data to plot and predict the survival ranking
X_test_subset = X_test[0:5]
y_test_subset = pd.DataFrame(y_test[0:5], index=X_test_subset.index)
df_test_subset = pd.concat([X_test_subset, y_test_subset], axis=1)

# Predict the indiviual survival ranking for the subset of observations
surv_pred = model.predict(X_test_subset)
