# Imports
from lifelines import KaplanMeierFitter
from lifelines.utils import qth_survival_times
import pandas as pd
from matplotlib import pyplot as plt

# Import data (already encoded)
df = pd.read_csv("../kidney_transplant.csv", header=0)
T = df["time"]
E = df["death"]

# Instantiate the kaplan meier estimator, with 95% confidence interval
# to estimate the survival function
model = KaplanMeierFitter(alpha=0.5)

# Fit the estimator
model.fit(T, event_observed=E)

# Median survival time does not exist, but the 80% does
print(f"Median survival time: {model.median_survival_time_} days.")
print(f"The 80% quantile: {qth_survival_times(0.8, model.survival_function_)} days.")

# Plot the survival function for the whole population
# (you can see that the median does not exist)
model.plot_survival_function(at_risk_counts=True)
plt.title("Survival function with 95% confidence interval")
plt.tight_layout()
plt.show()
