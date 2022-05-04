# Imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Import data
df = pd.read_csv("../digits.csv", header=0)
y = df["target"]
X = df.drop(columns=["target"])

# Standardizing features with the standard score
# Very important, so that PCA works well here
scaler = StandardScaler(with_mean=True, with_std=True)
X_scaled = scaler.fit_transform(X)

# Fit the model and keep all components
model = PCA()
components = model.fit_transform(X_scaled)

# Build dataset with the components and the digit labels
components_labels = pd.DataFrame(
    components,
    columns=[f"Component {i}" for i in range(1, components.shape[1]+1)]
)
components_labels["y"] = y

# Get explained variance of the first two components
explained_variance_first_two = np.round(sum(model.explained_variance_ratio_[:2]*100), 2)
print(f"Percentage of variance explained of the first two components: {explained_variance_first_two}%")

# Get number of components explaining at least 90% of the variance
n_components_90 = np.min(np.where((np.cumsum(model.explained_variance_ratio_) > 0.90))) + 1
print(f"Number of components explaining at least 90% of the variance: {n_components_90}")

# Visualize cluster as a scatter plot (only first 2 components)
plt.figure(figsize=(15, 15))
sns.scatterplot(
    x="Component 1", y="Component 2", hue="y",
    palette=sns.color_palette("hls", len(set(y))),
    data=components_labels,
    legend="full",
)
plt.title(f"PC1 and PC2 of the Principal Component Analysis")
plt.show()
