import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Transactional Anomaly Detection Framework",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Transactional Anomaly Detection Engine")
st.markdown("Advanced clustering & unsupervised learning pipeline for financial fraud and outlier detection.")

# ---------------------------------------------------------
# Sidebar: Dataset Selection & Controls
# ---------------------------------------------------------
st.sidebar.header("⚙️ Data & Model Controls")

data_source = st.sidebar.selectbox("Transaction Data Source", ["Synthetic Financial Log (PaySim-Style)", "Upload Custom CSV"])

@st.cache_data
def generate_synthetic_transactions(records=2000):
    np.random.seed(42)
    tx_types = np.random.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"], size=records, p=[0.4, 0.25, 0.25, 0.1])
    amounts = np.random.exponential(scale=250, size=records) + 10
    old_bal = np.random.uniform(500, 50000, size=records)
    new_bal = np.maximum(0, old_bal - amounts)
    hour = np.random.randint(0, 24, size=records)
    location_risk = np.random.uniform(0.01, 0.3, size=records)
    
    df = pd.DataFrame({
        "Transaction_ID": [f"TX-{100000 + i}" for i in range(records)],
        "Type": tx_types,
        "Amount": np.round(amounts, 2),
        "Old_Balance": np.round(old_bal, 2),
        "New_Balance": np.round(new_bal, 2),
        "Hour_of_Day": hour,
        "Location_Risk_Score": np.round(location_risk, 3)
    })
    
    # Inject Synthetic Fraud/Anomalies (3% contamination)
    anomaly_idx = np.random.choice(records, size=int(records * 0.03), replace=False)
    df.loc[anomaly_idx, "Amount"] *= np.random.uniform(8, 20, size=len(anomaly_idx))
    df.loc[anomaly_idx, "Old_Balance"] = df.loc[anomaly_idx, "Amount"] * 1.1
    df.loc[anomaly_idx, "New_Balance"] = 0
    df.loc[anomaly_idx, "Location_Risk_Score"] = np.random.uniform(0.75, 0.99, size=len(anomaly_idx))
    
    return df

if data_source == "Synthetic Financial Log (PaySim-Style)":
    num_samples = st.sidebar.slider("Synthetic Sample Size", 500, 5000, 2000, 500)
    raw_df = generate_synthetic_transactions(num_samples)
else:
    uploaded_file = st.sidebar.file_uploader("Upload Transaction CSV", type=["csv"])
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
    else:
        st.info("Please upload a CSV file. Falling back to synthetic data for now.")
        raw_df = generate_synthetic_transactions(2000)

df = raw_df.copy()

# ---------------------------------------------------------
# Sidebar: Algorithm Configuration
# ---------------------------------------------------------
st.sidebar.subheader("🤖 Clustering Model Selection")
algorithm = st.sidebar.selectbox("Algorithm", ["DBSCAN (Density-Based)", "K-Means Distance Thresholding", "Isolation Forest"])

if algorithm == "DBSCAN (Density-Based)":
    eps = st.sidebar.slider("Epsilon (eps)", 0.2, 3.0, 0.85, 0.05)
    min_samples = st.sidebar.slider("Min Samples", 2, 25, 8)
elif algorithm == "K-Means Distance Thresholding":
    n_clusters = st.sidebar.slider("Clusters (K)", 2, 8, 3)
    percentile = st.sidebar.slider("Outlier Cutoff Percentile", 80, 99, 95)
else: # Isolation Forest
    contamination = st.sidebar.slider("Contamination Factor", 0.01, 0.15, 0.03, 0.01)

# ---------------------------------------------------------
# Preprocessing Pipeline
# ---------------------------------------------------------
feature_cols = ["Amount", "Old_Balance", "New_Balance", "Hour_of_Day", "Location_Risk_Score"]

# One-Hot Encode Categorical Features if present
if "Type" in df.columns:
    df_encoded = pd.get_dummies(df[feature_cols + ["Type"]], columns=["Type"], drop_first=True)
else:
    df_encoded = df[feature_cols].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

# ---------------------------------------------------------
# Clustering Execution
# ---------------------------------------------------------
if algorithm == "DBSCAN (Density-Based)":
    model = DBSCAN(eps=eps, min_samples=min_samples)
    df["Cluster"] = model.fit_predict(X_scaled)
    df["Is_Anomaly"] = df["Cluster"] == -1

elif algorithm == "K-Means Distance Thresholding":
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_
    distances = np.linalg.norm(X_scaled - centroids[df["Cluster"]], axis=1)
    threshold = np.percentile(distances, percentile)
    df["Is_Anomaly"] = distances > threshold

else: # Isolation Forest
    iso = IsolationForest(contamination=contamination, random_state=42)
    labels = iso.fit_predict(X_scaled)
    df["Cluster"] = labels
    df["Is_Anomaly"] = df["Cluster"] == -1

# PCA 2D Dimensionality Reduction for Visuals
pca = PCA(n_components=2)
pca_coords = pca.fit_transform(X_scaled)
df["PCA_1"] = pca_coords[:, 0]
df["PCA_2"] = pca_coords[:, 1]

# ---------------------------------------------------------
# Dashboard Layout & Metrics
# ---------------------------------------------------------
st.header("📊 Analytical Overview")

anomalies_count = int(df["Is_Anomaly"].sum())
total_records = len(df)
anomaly_rate = (anomalies_count / total_records) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Transactions", f"{total_records:,}")
c2.metric("Flagged Outliers", f"{anomalies_count:,}")
c3.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")

if len(set(df["Cluster"])) > 1:
    sil_score = silhouette_score(X_scaled, df["Cluster"])
    c4.metric("Silhouette Score", f"{sil_score:.3f}")
else:
    c4.metric("Silhouette Score", "N/A (1 Cluster)")

st.markdown("---")

# ---------------------------------------------------------
# Visualizations
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📉 PCA Outlier Cluster Projection", "📊 Feature Distribution Analysis", "🔍 Anomaly Inspector Table"])

with tab1:
    df["Status"] = df["Is_Anomaly"].map({True: "Anomaly", False: "Normal"})
    
    fig_pca = px.scatter(
        df,
        x="PCA_1",
        y="PCA_2",
        color="Status",
        symbol="Status",
        color_discrete_map={"Normal": "#3B82F6", "Anomaly": "#EF4444"},
        hover_data=["Transaction_ID", "Amount", "Type", "Location_Risk_Score"],
        title="2D Principal Component Analysis (PCA) Cluster Projection",
        template="plotly_dark"
    )
    fig_pca.update_traces(marker=dict(size=8, opacity=0.8))
    st.plotly_chart(fig_pca, use_container_width=True)

with tab2:
    st.subheader("Normal vs. Anomalous Feature Comparison")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        fig_box = px.box(
            df,
            x="Status",
            y="Amount",
            color="Status",
            color_discrete_map={"Normal": "#3B82F6", "Anomaly": "#EF4444"},
            title="Transaction Amount Spread ($)",
            template="plotly_dark"
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
    with col_b:
        fig_risk = px.histogram(
            df,
            x="Location_Risk_Score",
            color="Status",
            barmode="overlay",
            color_discrete_map={"Normal": "#3B82F6", "Anomaly": "#EF4444"},
            title="Location Risk Score Distribution",
            template="plotly_dark"
        )
        st.plotly_chart(fig_risk, use_container_width=True)

with tab3:
    st.subheader("Flagged Transactions Log")
    
    search_term = st.text_input("🔍 Search by Transaction ID or Type", "")
    
    filtered_df = df[df["Is_Anomaly"] == True]
    if search_term:
        filtered_df = filtered_df[
            filtered_df["Transaction_ID"].astype(str).str.contains(search_term, case=False) |
            filtered_df["Type"].astype(str).str.contains(search_term, case=False)
        ]
        
    st.dataframe(
        filtered_df[["Transaction_ID", "Type", "Amount", "Old_Balance", "New_Balance", "Hour_of_Day", "Location_Risk_Score", "Cluster", "Status"]],
        use_container_width=True
    )
    
    # Export CSV Option
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Flagged Anomalies CSV",
        data=csv,
        file_name="flagged_anomalies.csv",
        mime="text/csv"
    )