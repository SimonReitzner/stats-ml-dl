# Imports
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt

# Import data (already encoded)
df = pd.read_csv("../kidney_transplant.csv", header=0)
y = Surv().from_dataframe("death", "time", df)
X = df.drop(columns=["death", "time"])

# Split data into train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=0
)

# Create grid of parameters to tune
param_grid = [
    {
        "n_estimators": [10, 50, 100, 1000],
        "min_samples_split": [5, 10, 15],
        "min_samples_leaf" : [5, 10, 15],
        "max_features" : ["auto", "sqrt", "log2", None]
    },
 ]

# Definition of a simple scoring function for grid search
def scoring_cindex(model, X, y):
    """Return concordance_index_censored"""
    prediction = model.predict(X)
    result = concordance_index_censored(y["death"], y["time"], prediction)
    return result[0]

# Instantiate a random survival forest, a shufflesplit and a grid search
estimator = RandomSurvivalForest(random_state=0)
cv = ShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
gscv = GridSearchCV(estimator, param_grid,
    scoring=scoring_cindex,
    n_jobs=3,
    refit=True,
    cv=cv,
    verbose=2
)

# Tune and fit the model (refit=True)
gscv.fit(X_train, y_train)

# Get the test score
model = gscv.best_estimator_
print(f"Test score of the tuned model: {model.score(X_test, y_test)}")

# Build a subset of the test data to plot and predict the survival function
X_test_subset = X_test[0:5]
y_test_subset = pd.DataFrame(y_test[0:5], index=X_test_subset.index)
df_test_subset = pd.concat([X_test_subset, y_test_subset], axis=1)

# Predict the indiviual survival function for the subset of observations
surv_pred = model.predict_survival_function(X_test_subset, return_array=True)

# Plot the survival functions
for i, s in enumerate(surv_pred):
    plt.step(model.event_times_, s, where="post", label=str(i))
plt.ylabel("Survival probability")
plt.xlabel("Time in days")
plt.legend()
plt.grid(True)
plt.show()