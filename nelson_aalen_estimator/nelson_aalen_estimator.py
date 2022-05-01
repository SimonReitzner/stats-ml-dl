# Imports
from lifelines import NelsonAalenFitter
import pandas as pd
from matplotlib import pyplot as plt

# Import data (already encoded)
df = pd.read_csv("../kidney_transplant.csv", header=0)
T = df["time"]
E = df["death"]

# Instantiate the nelson aalen estimator, with 95% confidence interval
# to estimate the cumulative hazard function
model = NelsonAalenFitter(alpha=0.5, nelson_aalen_smoothing=False)

# Fit the estimator
model.fit(T, event_observed=E)

# Plot the cumulative hazard function for the whole population
# High hazard rates at the beginning
model.plot_cumulative_hazard(at_risk_counts=True)
plt.title("Cumulative hazard function with 95% confidence interval")
plt.tight_layout()
plt.show()
