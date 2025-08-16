import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load Data
spectral = pd.read_csv("20231225_dfall_obs_data_and_spectral_features_revision1_n469.csv")
playback = pd.read_csv("20230409_playback_data_for_upload.csv", encoding="latin1")

# Merge on CallerID (spectral) ↔ Caller (playback)
data = spectral.merge(playback, left_on="CallerID", right_on="Caller", how="inner")

# Use LatApr (latency to approach) as target
data = data.dropna(subset=["LatApr"])

# STEP 0: Inspect numeric features
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
print("Total numeric features:", len(numeric_cols))

# A1:Entropy
def equal_width_binning(series, bins=4):
    return pd.cut(series, bins=bins, labels=False)

def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

# Bin LatApr into 4 categories
y_binned = equal_width_binning(data["LatApr"], bins=4)
print("Entropy of LatApr:", entropy(y_binned))

#A2: Gini Index
def gini_index(y):
    values, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.sum(probabilities**2)

print("Gini Index of LatApr:", gini_index(y_binned))

#A3–A4: Root Node by Info Gain
X = data[numeric_cols].drop(columns=["LatApr"], errors="ignore")
y = y_binned

info_gain = mutual_info_classif(X.fillna(0), y)
root_feature = X.columns[np.argmax(info_gain)]
print("Best Root Feature (by Info Gain):", root_feature)

#A5: Build Decision Tree
dt = DecisionTreeClassifier(max_depth=3, criterion="entropy", random_state=42)
dt.fit(X.fillna(0), y)

#A6: Visualize Tree
plt.figure(figsize=(20,10))
plot_tree(dt, feature_names=X.columns, class_names=[str(c) for c in np.unique(y)], filled=True)
plt.show()

# A7: Decision Boundary with 2 Best Features
# Rank features by information gain
feature_ranking = sorted(zip(X.columns, info_gain), key=lambda x: x[1], reverse=True)

# Pick top 2
feature1, feature2 = feature_ranking[0][0], feature_ranking[1][0]
print(f"Using features for A7: {feature1}, {feature2}")

X2 = data[[feature1, feature2]].fillna(0)
y2 = y

X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.3, random_state=42)

dt2 = DecisionTreeClassifier(max_depth=3)
dt2.fit(X_train, y_train)

# Meshgrid for decision boundary
x_min, x_max = X2.iloc[:, 0].min() - 1, X2.iloc[:, 0].max() + 1
y_min, y_max = X2.iloc[:, 1].min() - 1, X2.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Convert mesh to DataFrame with feature names
grid_df = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=[feature1, feature2])

Z = dt2.predict(grid_df)
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X2.iloc[:, 0], X2.iloc[:, 1], c=y2, s=20, edgecolor='k')
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.title(f"Decision Boundary (DT with {feature1} & {feature2})")
plt.show()
