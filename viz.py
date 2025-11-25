import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

def load_retriever_data(file_paths):
    """Load data from multiple retriever JSON files"""
    retrievers = {}
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)
            retriever_name = data['retriever_name']
            retrievers[retriever_name] = data
    return retrievers

def plot_aggregated_metrics(retrievers, output_path='aggregated_comparison.png'):
    """Plot aggregated metrics comparison across retrievers"""
    metrics = ['mrr', 'precision@1', 'precision@3', 'precision@5', 'precision@10', 
               'recall@1', 'recall@3', 'recall@5', 'recall@10',
               'ndcg@1', 'ndcg@3', 'ndcg@5', 'ndcg@10',
               'hit_rate@1', 'hit_rate@3', 'hit_rate@5', 'hit_rate@10']
    
    # Prepare data
    data = []
    for retriever_name, retriever_data in retrievers.items():
        agg_metrics = retriever_data['aggregated_metrics']
        for metric in metrics:
            if metric in agg_metrics:
                data.append({
                    'Retriever': retriever_name,
                    'Metric': metric,
                    'Value': agg_metrics[metric]
                })
    
    df = pd.DataFrame(data)
    
    # Create subplots for different metric categories
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # MRR and Metadata Consistency
    ax1 = axes[0, 0]
    metrics_subset = ['mrr', 'metadata_consistency']
    df_subset = df[df['Metric'].isin(metrics_subset)]
    df_pivot = df_subset.pivot(index='Retriever', columns='Metric', values='Value')
    df_pivot.plot(kind='bar', ax=ax1)
    ax1.set_title('MRR and Metadata Consistency', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Score')
    ax1.set_xlabel('Retriever')
    ax1.legend(title='Metric')
    ax1.tick_params(axis='x', rotation=45)
    
    # Precision metrics
    ax2 = axes[0, 1]
    precision_metrics = [m for m in metrics if m.startswith('precision@')]
    df_precision = df[df['Metric'].isin(precision_metrics)]
    df_pivot = df_precision.pivot(index='Retriever', columns='Metric', values='Value')
    df_pivot.plot(kind='bar', ax=ax2)
    ax2.set_title('Precision Metrics', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Score')
    ax2.set_xlabel('Retriever')
    ax2.legend(title='Metric')
    ax2.tick_params(axis='x', rotation=45)
    
    # Recall metrics
    ax3 = axes[1, 0]
    recall_metrics = [m for m in metrics if m.startswith('recall@')]
    df_recall = df[df['Metric'].isin(recall_metrics)]
    df_pivot = df_recall.pivot(index='Retriever', columns='Metric', values='Value')
    df_pivot.plot(kind='bar', ax=ax3)
    ax3.set_title('Recall Metrics', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Score')
    ax3.set_xlabel('Retriever')
    ax3.legend(title='Metric')
    ax3.tick_params(axis='x', rotation=45)
    
    # NDCG metrics
    ax4 = axes[1, 1]
    ndcg_metrics = [m for m in metrics if m.startswith('ndcg@')]
    df_ndcg = df[df['Metric'].isin(ndcg_metrics)]
    df_pivot = df_ndcg.pivot(index='Retriever', columns='Metric', values='Value')
    df_pivot.plot(kind='bar', ax=ax4)
    ax4.set_title('NDCG Metrics', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Score')
    ax4.set_xlabel('Retriever')
    ax4.legend(title='Metric')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved aggregated metrics plot to {output_path}")
    plt.close()

def plot_key_metrics_comparison(retrievers, output_path='key_metrics_comparison.png'):
    """Plot key metrics in a single comparison chart"""
    key_metrics = ['mrr', 'ndcg@10', 'recall@10', 'precision@10', 'hit_rate@10']
    
    data = []
    for retriever_name, retriever_data in retrievers.items():
        agg_metrics = retriever_data['aggregated_metrics']
        for metric in key_metrics:
            if metric in agg_metrics:
                data.append({
                    'Retriever': retriever_name,
                    'Metric': metric,
                    'Value': agg_metrics[metric]
                })
    
    df = pd.DataFrame(data)
    df_pivot = df.pivot(index='Retriever', columns='Metric', values='Value')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    df_pivot.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Key Retrieval Metrics Comparison', fontsize=16, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Retriever', fontsize=12)
    ax.legend(title='Metric', fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved key metrics comparison to {output_path}")
    plt.close()

def plot_query_level_metrics(retrievers, output_path='query_level_metrics.png'):
    """Plot per-query metrics across retrievers"""
    # Focus on key metrics at query level
    metrics_to_plot = ['mrr', 'ndcg@10', 'recall@10']
    
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(14, 12))
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        # Prepare data
        data = []
        for retriever_name, retriever_data in retrievers.items():
            query_metrics = retriever_data['query_metrics']
            for query_id, query_data in query_metrics.items():
                if metric in query_data:
                    data.append({
                        'Retriever': retriever_name,
                        'Query': query_id,
                        'Value': query_data[metric]
                    })
        
        df = pd.DataFrame(data)
        df_pivot = df.pivot(index='Query', columns='Retriever', values='Value')
        
        df_pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title(f'{metric.upper()} per Query', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=11)
        ax.set_xlabel('Query ID', fontsize=11)
        ax.legend(title='Retriever', fontsize=9)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved query-level metrics to {output_path}")
    plt.close()

def plot_heatmap(retrievers, output_path='retriever_heatmap.png'):
    """Create a heatmap showing performance across all metrics"""
    metrics = ['mrr', 'precision@5', 'precision@10', 'recall@5', 'recall@10',
               'ndcg@5', 'ndcg@10', 'hit_rate@5', 'hit_rate@10', 'metadata_consistency']
    
    data = []
    for retriever_name, retriever_data in retrievers.items():
        row = {'Retriever': retriever_name}
        agg_metrics = retriever_data['aggregated_metrics']
        for metric in metrics:
            if metric in agg_metrics:
                row[metric] = agg_metrics[metric]
        data.append(row)
    
    df = pd.DataFrame(data)
    df = df.set_index('Retriever')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax, 
                cbar_kws={'label': 'Score'}, linewidths=0.5)
    ax.set_title('Retriever Performance Heatmap', fontsize=16, fontweight='bold')
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Retrievers', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {output_path}")
    plt.close()

def create_summary_table(retrievers, output_path='summary_table.csv'):
    """Create a summary table with all aggregated metrics"""
    data = []
    for retriever_name, retriever_data in retrievers.items():
        row = {'Retriever': retriever_name}
        row.update(retriever_data['aggregated_metrics'])
        data.append(row)
    
    df = pd.DataFrame(data)
    df = df.set_index('Retriever')
    df = df.round(4)
    
    df.to_csv(output_path)
    print(f"Saved summary table to {output_path}")
    
    return df

def main():
    # You can modify this list to include your JSON file paths
    file_paths = [
        'retrieval_output/run_1764103419/evaluation/Content_(recursive)_metrics.json',
        'retrieval_output/run_1764103419/evaluation/Content_(naive)_metrics.json',
        "retrieval_output/run_1764103419/evaluation/Content_(semantic)_metrics.json",
        'retrieval_output/run_1764103419/evaluation/Prefix-Fusion_(naive)_metrics.json',
        'retrieval_output/run_1764103419/evaluation/Prefix-Fusion_(recursive)_metrics.json',
        'retrieval_output/run_1764103419/evaluation/Prefix-Fusion_(semantic)_metrics.json',
        'retrieval_output/run_1764103419/evaluation/Reranker-Prefix_(naive)_metrics.json',
        "retrieval_output/run_1764103419/evaluation/Reranker-Prefix_(recursive)_metrics.json",
        "retrieval_output/run_1764103419/evaluation/Reranker-Prefix_(semantic)_metrics.json",
        "retrieval_output/run_1764103419/evaluation/Reranker-TFIDF_(naive)_metrics.json",
        "retrieval_output/run_1764103419/evaluation/Reranker-TFIDF_(recursive)_metrics.json",
        "retrieval_output/run_1764103419/evaluation/Reranker-TFIDF_(semantic)_metrics.json",
    ]
    
    # Filter for files that exist
    existing_files = [f for f in file_paths if Path(f).exists()]
    
    if not existing_files:
        print("No JSON files found. Please update the file_paths list in the script.")
        print("Looking for files in current directory...")
        # Try to find JSON files automatically
        json_files = list(Path('.').glob('*.json'))
        if json_files:
            print(f"Found {len(json_files)} JSON files:")
            for f in json_files:
                print(f"  - {f}")
            existing_files = [str(f) for f in json_files]
        else:
            return
    
    print(f"Loading {len(existing_files)} retriever files...")
    retrievers = load_retriever_data(existing_files)
    print(f"Loaded retrievers: {', '.join(retrievers.keys())}")
    
    # Create output directory
    output_dir = Path('retriever_comparison_plots')
    output_dir.mkdir(exist_ok=True)
    
    # Generate all plots
    print("\nGenerating comparison plots...")
    plot_key_metrics_comparison(retrievers, output_dir / 'key_metrics_comparison.png')
    plot_aggregated_metrics(retrievers, output_dir / 'aggregated_comparison.png')
    plot_query_level_metrics(retrievers, output_dir / 'query_level_metrics.png')
    plot_heatmap(retrievers, output_dir / 'retriever_heatmap.png')
    
    # Create summary table
    print("\nGenerating summary table...")
    summary_df = create_summary_table(retrievers, output_dir / 'summary_table.csv')
    
    print("\n" + "="*60)
    print("Summary Table:")
    print("="*60)
    print(summary_df)
    print("\n" + "="*60)
    print(f"All outputs saved to {output_dir}/")
    print("="*60)

if __name__ == "__main__":
    main()