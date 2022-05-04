# Imports
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import data (already encoded)
df = pd.read_csv("../digits.csv", header=0)
y = df["target"]
X = df.drop(columns=["target"])

# Standardizing features with the standard score
scaler = StandardScaler(with_mean=True, with_std=True)
X_scaled = scaler.fit_transform(X)

# Fit the model with two dimensions
model = TSNE(n_components=2, learning_rate="auto", init="random")
embedding = model.fit_transform(X_scaled)

# Build dataset with the components and the digit labels
embedding_labels = pd.DataFrame(
    embedding,
    columns=[f"Dimension {i}" for i in range(1, embedding.shape[1]+1)]
)
embedding_labels["y"] = y

# Visualize cluster as scatter plot
plt.figure(figsize=(15, 15))
sns.scatterplot(
    x="Dimension 1", y="Dimension 2", hue="y",
    palette=sns.color_palette("hls", len(set(y))),
    data=embedding_labels,
    legend="full"
)
plt.title(f"t-distributed stochastic neighbor embedding with two dimensions")
plt.show()
