import pandas as pd
import numpy as np
import os
import joblib
from joblib import parallel_backend
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Change this to match your actual physical core count


# Load the dataset
def load_collatz_data(data_dir="data/analysis"):
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    df_list = [pd.read_csv(os.path.join(data_dir, file)) for file in all_files]
    df = pd.concat(df_list, ignore_index=True)
    
    # Convert Log2 values from string to lists of floats
    df["Log2 Values"] = df["Log2 Values"].apply(eval)  # Convert string representation to list
    
    # Feature Engineering
    df["Max Log2"] = df["Log2 Values"].apply(max)
    df["Min Log2"] = df["Log2 Values"].apply(min)
    df["Mean Log2"] = df["Log2 Values"].apply(np.mean)
    df["Variance Log2"] = df["Log2 Values"].apply(np.var)
    
    return df.drop(columns=["Log2 Values", "Collatz Sequence"])  # Remove sequence data for ML

# Supervised Learning - Regression
def train_models(df):
    features = ["Number", "Odd Steps", "Even Steps", "Odd Descents", "Max Log2", "Min Log2", "Mean Log2", "Variance Log2"]
    target = "Steps"
    
    X = df[features]
    y = df[target]
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler
    joblib.dump(scaler, "data/analysis/ML/scaler.pkl")
    print("Scaler saved.")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train models using parallel backend
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    with parallel_backend('threading', n_jobs=-1):  # Ensures better Windows compatibility
        for name, model in models.items():
            model.fit(X_train, y_train)
            joblib.dump(model, f"data/analysis/ML/{name}_model.pkl")
            print(f"{name} trained and saved.")
    
    return models, scaler

# Unsupervised Learning - Clustering
def perform_clustering(df):
    features = ["Odd Steps", "Even Steps", "Odd Descents", "Max Log2", "Min Log2", "Mean Log2", "Variance Log2"]
    X = df[features]
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Clustering using KMeans
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X_pca)
    
    # Visualization
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.5)
    
    # Add labels for clusters
    for i in range(kmeans.n_clusters):
        plt.text(kmeans.cluster_centers_[i, 0], kmeans.cluster_centers_[i, 1], str(i), fontsize=12, weight='bold', color='red')
    
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Clustering of Collatz Data")
    plt.colorbar()
    plt.show()
    
    return kmeans, pca

# Main execution
if __name__ == "__main__":
    df = load_collatz_data()
    models, scaler = train_models(df)
    kmeans, pca = perform_clustering(df)
    
    print("Machine Learning models trained and clustering performed successfully.")
