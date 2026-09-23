"""Standalone Gradio app for the Customer Segmentation project.

Loads the fitted artifacts (KMeans, StandardScaler, DBSCAN) plus the chosen settings
from the models/ folder that the training notebook saved — with zero notebook dependency.
The app exposes the same interactive explorer as the notebook's Section-13 demo.

Run standalone locally from the task root directory with:
    python app.py
"""

import io
import json
import os
import tempfile

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gradio as gr
from sklearn.cluster import KMeans


def _base_dir():
    """Return the directory holding this script (where the dataset + models/ live)."""
    return os.path.dirname(os.path.abspath(__file__))


# ---- Load the dataset + the saved artifacts (single source of truth) ----
df = pd.read_excel(os.path.join(_base_dir(), 'Mall_Customers.xlsx'))
df.columns = df.columns.str.strip()

scaler = joblib.load(os.path.join(_base_dir(), 'models', 'scaler.pkl'))
kmeans = joblib.load(os.path.join(_base_dir(), 'models', 'kmeans.pkl'))
dbscan = joblib.load(os.path.join(_base_dir(), 'models', 'dbscan.pkl'))
with open(os.path.join(_base_dir(), 'models', 'params.json')) as f:
    params = json.load(f)

FEATURE_COLUMNS = params['feature_columns']     # ['Annual Income (k$)', 'Spending Score (1-100)']
OPTIMAL_K = params['optimal_k']                 # silhouette-selected k (=5)
DBSCAN_EPS = params['dbscan_eps']
DBSCAN_MIN_SAMPLES = params['dbscan_min_samples']


def _scaled_features():
    """Scale the two selected features with the SAVED scaler (matches training)."""
    X = df[FEATURE_COLUMNS].values
    return scaler.transform(X)


def run_clustering(algorithm, n_clusters):
    """Main function called by the Gradio interface.

    Args:
        algorithm: 'K-Means' or 'DBSCAN'
        n_clusters: Number of clusters (used only for K-Means; default = saved optimal k)

    Returns:
        scatter_plot: Matplotlib figure with cluster scatter plot
        summary_table: DataFrame with cluster averages
        spending_chart: Matplotlib figure with average spending bar chart
    """
    X_scaled = _scaled_features()

    # --- Step 1: obtain labels for the selected algorithm ---
    if algorithm == 'K-Means':
        if int(n_clusters) == OPTIMAL_K:
            # Reuse the saved, silhouette-selected model when the default k is chosen.
            model = kmeans
            labels = kmeans.labels_
        else:
            # Interactive exploration: refit K-Means for the user-chosen k on the SAME
            # scaled features (same random_state/n_init as the notebook).
            model = KMeans(n_clusters=int(n_clusters), random_state=42, n_init=10)
            labels = model.fit_predict(X_scaled)
    else:
        # DBSCAN is deterministic; the saved model was fit on the exact same scaled data.
        model = dbscan
        labels = dbscan.labels_

    # --- Step 2: Create Scatter Plot ---
    fig_scatter, ax_scatter = plt.subplots(figsize=(9, 6))
    unique_labels = sorted(set(labels))
    plot_colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1)))

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            # Noise points (DBSCAN only): gray with 'x' marker
            ax_scatter.scatter(df.loc[mask, FEATURE_COLUMNS[0]],
                               df.loc[mask, FEATURE_COLUMNS[1]],
                               c='gray', marker='x', s=60, alpha=0.5, label='Noise')
        else:
            ax_scatter.scatter(df.loc[mask, FEATURE_COLUMNS[0]],
                               df.loc[mask, FEATURE_COLUMNS[1]],
                               c=[plot_colors[idx % len(plot_colors)]],
                               s=80, alpha=0.7, edgecolors='white', linewidths=0.5,
                               label=f'Cluster {label}')

    # Plot centroids for K-Means (transform them back to the original scale)
    if algorithm == 'K-Means':
        centroids_orig = scaler.inverse_transform(model.cluster_centers_)
        ax_scatter.scatter(centroids_orig[:, 0], centroids_orig[:, 1],
                           c='black', marker='X', s=300, linewidths=2,
                           edgecolors='white', label='Centroids', zorder=5)

    ax_scatter.set_title(f'{algorithm} - Customer Segments', fontsize=14, fontweight='bold')
    ax_scatter.set_xlabel(FEATURE_COLUMNS[0], fontsize=12)
    ax_scatter.set_ylabel(FEATURE_COLUMNS[1], fontsize=12)
    ax_scatter.legend(fontsize=10, loc='best')
    ax_scatter.grid(True, alpha=0.2)
    plt.tight_layout()

    # --- Step 3: Create Summary Table ---
    temp_df = df.copy()
    temp_df['Cluster'] = labels
    summary = temp_df.groupby('Cluster')[['Age', *FEATURE_COLUMNS]].mean().round(2)
    summary['Count'] = temp_df.groupby('Cluster').size().values
    summary = summary[['Count', 'Age', *FEATURE_COLUMNS]]
    summary.columns = ['Count', 'Avg Age', 'Avg Income (k$)', 'Avg Spending Score']
    summary = summary.reset_index()

    # --- Step 4: Create Average Spending Bar Chart ---
    fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
    avg_sp = temp_df.groupby('Cluster')[FEATURE_COLUMNS[1]].mean().sort_values(ascending=True)
    bar_colors = plt.cm.tab10(np.linspace(0, 1, len(avg_sp)))
    bars = ax_bar.barh(avg_sp.index.astype(str), avg_sp.values,
                       color=bar_colors, edgecolor='white', height=0.6)
    for bar, val in zip(bars, avg_sp.values):
        ax_bar.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}', va='center', fontsize=10, fontweight='bold')
    ax_bar.set_title('Avg Spending Score per Cluster', fontsize=13, fontweight='bold')
    ax_bar.set_xlabel('Average Spending Score', fontsize=11)
    ax_bar.set_ylabel('Cluster', fontsize=11)
    ax_bar.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    return fig_scatter, summary, fig_bar


# --- Build the Gradio Interface with gr.Blocks (mirrors the notebook demo) ---
with gr.Blocks(title="Customer Segmentation Explorer") as demo:
    gr.Markdown("""
    # Customer Segmentation Explorer

    Select an algorithm and adjust parameters to explore different customer segments.

    - **K-Means:** Choose the number of clusters with the slider
    - **DBSCAN:** Uses the saved model (eps=%.2f, min_samples=%d) - no cluster count needed
    """ % (DBSCAN_EPS, DBSCAN_MIN_SAMPLES))

    with gr.Row():
        with gr.Column(scale=1):
            algorithm_selector = gr.Radio(
                choices=['K-Means', 'DBSCAN'],
                value='K-Means',
                label='Clustering Algorithm',
                info='Select K-Means or DBSCAN'
            )
            cluster_slider = gr.Slider(
                minimum=2, maximum=10,
                value=float(OPTIMAL_K),
                step=1,
                label=f'Number of Clusters (K-Means only) - Default: {OPTIMAL_K}',
                info='Adjust the number of clusters for K-Means'
            )
            run_button = gr.Button("Run Clustering", variant="primary", size="lg")

        with gr.Column(scale=2):
            scatter_output = gr.Image(label="Cluster Scatter Plot", type="filepath")
            table_output = gr.Dataframe(label="Cluster Summary")
            bar_output = gr.Image(label="Average Spending per Cluster", type="filepath")

    def run_and_return_images(algorithm, n_clusters):
        """Wrapper that runs clustering and saves plots as images for Gradio."""
        fig_scatter, summary, fig_bar = run_clustering(algorithm, n_clusters)

        scatter_path = os.path.join(tempfile.gettempdir(), 'scatter_plot.png')
        fig_scatter.savefig(scatter_path, dpi=100, bbox_inches='tight')
        plt.close(fig_scatter)

        bar_path = os.path.join(tempfile.gettempdir(), 'bar_chart.png')
        fig_bar.savefig(bar_path, dpi=100, bbox_inches='tight')
        plt.close(fig_bar)

        return scatter_path, summary, bar_path

    run_button.click(
        fn=run_and_return_images,
        inputs=[algorithm_selector, cluster_slider],
        outputs=[scatter_output, table_output, bar_output]
    )

# Local launch (no share/inbrowser flags here; add share=True for a public URL).
if __name__ == '__main__':
    demo.launch(theme=gr.themes.Soft())