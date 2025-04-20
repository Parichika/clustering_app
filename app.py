#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 00:21:28 2025

@author: parichikaphumikakrak
"""

import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from matplotlib.patches import Circle


# Load model
with open('kmeans_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
    
    
# Page title
st.title("🔍 K-Means Clustering App with Iris Dataset by Parichika Phumikakrak")


# Set page config
st.set_page_config(page_title="K-Means Clustering", layout="centered")


# Display dataset
X, _ = make_blobs(n_samples=300, centers=loaded_model.n_clusters, cluster_std=0.60, random_state=0)


# Predict using the loaded model
y_kmeans = loaded_model.predict(X)


# Plotting
fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')


# Plot centroids as big red circles
centroids = loaded_model.cluster_centers_
ax.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', label='Centroids')  # marker is 'o' by default
ax.set_title('k-Means Clustering')
ax.legend()
st.pyplot(fig)
