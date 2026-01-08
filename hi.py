import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Title
st.title("Hierarchical Clustering – Iris Dataset")

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

st.subheader("Dataset Preview (Iris)")
st.dataframe(df.head())

# Sidebar options
st.sidebar.header("Clustering Settings")
n_clusters = st.sidebar.slider("Number of Clusters", 2, 5, 3)

# Hierarchical clustering
model = AgglomerativeClustering(n_clusters=n_clusters)
labels = model.fit_predict(df)

# Cluster visualization (using first two features)
st.subheader("Cluster Visualization")
fig, ax = plt.subplots()
ax.scatter(
    df.iloc[:, 0],
    df.iloc[:, 1],
    c=labels
)
ax.set_xlabel(df.columns[0])
ax.set_ylabel(df.columns[1])
st.pyplot(fig)

# Dendrogram
st.subheader("Dendrogram")
linked = linkage(df, method="ward")
fig2, ax2 = plt.subplots(figsize=(10, 4))
dendrogram(linked, truncate_mode="lastp", p=30, ax=ax2)
st.pyplot(fig2)
