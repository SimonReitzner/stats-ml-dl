# Imports
from calendar import LocaleHTMLCalendar
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import ParameterGrid, train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Import data
df = pd.read_csv("../breast_cancer.csv", header=0)
y = df["target"]
X = df.drop(columns=["target"])

# Standardizing features with the standard score
# and encoding the target
scaler = StandardScaler(with_mean=True, with_std=True)
X_prep = scaler.fit_transform(X)

encoder = LabelEncoder()
y_prep = encoder.fit_transform(y)

# Split data into train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X_prep, y_prep,
    test_size=0.25,
    random_state=0
)

# Create grid of parameters to tune
param_grid = [
    {
        "penalty" : ["l2"],
        "C" : [0.01, 0.1, 1, 10, 100],
        "fit_intercept" : [True, False],
        "solver" : ["newton-cg"],
    },
    {
        "penalty" : ["none"],
        "fit_intercept" : [True, False],
        "solver" : ["newton-cg"],  
    }
]

# Instantiate a logistic regression estimator, a use kfold and grid search
estimator = LogisticRegression(random_state=0)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)
gscv = GridSearchCV(estimator, param_grid,
    scoring="accuracy",
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

# Predict the label for the test set
y_pred = model.predict(X_test)

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)
disp.plot()
plt.show()