"""
Enhanced Plotting for Retrieval Evaluation Results
Creates publication/presentation-ready visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configuration
INPUT_CSV = r"C:\Users\moksh\Desktop\UICLaborDocsChatbot-clara_work\UICLaborDocsChatbot-clara_work\metarag\retrieval_output\run_1764103419\evaluation_summary_k10.csv"
OUTPUT_DIR = r"C:\Users\moksh\Desktop\UICLaborDocsChatbot-clara_work\UICLaborDocsChatbot-clara_work\metarag\retrieval_output\run_1764103419"
EVAL_K = 10

# Read data
df = pd.read_csv(INPUT_CSV)

# Shorten configuration names for better display
def shorten_config_name(name):
    """Simplify configuration names for visualization"""
    name = name.replace("Reranker-", "")
    name = name.replace("Content_", "Content-")
    name = name.replace("Prefix-Fusion_", "PrefixFusion-")
    name = name.replace("TF-IDF_", "TFIDF-")
    return name

df['Config_Short'] = df['filename'].apply(shorten_config_name)

# Sort by mrr
df = df.sort_values('mrr', ascending=False)

# ============================================================================
# PLOT 1: Top 8 Configurations - Grouped Bar Chart
# ============================================================================
def plot_top_configs_grouped():
    """Cleaner grouped bar chart focusing on top performers"""
    
    # Select top 8 configurations
    df_top = df.head(8).copy()
    
    # Prepare data for plotting
    metrics = ['mrr', f'hit_rate@{EVAL_K}', f'precision@{EVAL_K}', f'recall@{EVAL_K}']
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(df_top))
    width = 0.2
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for i, metric in enumerate(metrics):
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, df_top[metric], width, 
                      label=metric.replace(f'@{EVAL_K}', f'@{EVAL_K}'),
                      color=colors[i], alpha=0.9, edgecolor='white', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Configuration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title(f'Top 8 Retrieval Configurations (K={EVAL_K})', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df_top['Config_Short'], rotation=45, ha='right', fontsize=11)
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    output_path = Path(OUTPUT_DIR) / "top8_configurations_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")
    plt.close()

# ============================================================================
# PLOT 2: Heatmap with Better Design
# ============================================================================
def plot_heatmap_enhanced():
    """Enhanced heatmap with better color scheme and annotations"""
    
    metrics = ['mrr', f'hit_rate@{EVAL_K}', f'precision@{EVAL_K}', f'recall@{EVAL_K}', 'source_acc']
    df_heat = df.set_index('Config_Short')[metrics]
    
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Use a diverging colormap
    sns.heatmap(df_heat, annot=True, fmt='.3f', cmap='RdYlGn', 
                center=0.7, vmin=0.4, vmax=1.0,
                linewidths=2, linecolor='white',
                cbar_kws={'label': 'Score', 'shrink': 0.8},
                annot_kws={'fontsize': 10, 'fontweight': 'bold'})
    
    ax.set_title(f'Retrieval Performance Heatmap (K={EVAL_K})', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Metrics', fontsize=13, fontweight='bold')
    ax.set_ylabel('Configuration', fontsize=13, fontweight='bold')
    
    # Rotate y-axis labels for better readability
    plt.yticks(rotation=0, fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    plt.tight_layout()
    output_path = Path(OUTPUT_DIR) / "heatmap_enhanced.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")
    plt.close()

# ============================================================================
# PLOT 3: Separate Subplots for Each Metric
# ============================================================================
def plot_metric_subplots():
    """4 separate subplots, one for each metric"""
    
    metrics = ['mrr', f'hit_rate@{EVAL_K}', f'precision@{EVAL_K}', f'recall@{EVAL_K}']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[idx]
        
        # Sort by current metric for this subplot
        df_sorted = df.sort_values(metric, ascending=False)
        
        bars = ax.barh(df_sorted['Config_Short'], df_sorted[metric], 
                       color=color, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, df_sorted[metric])):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', 
                   va='center', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(metric, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlim(0, 1.1)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Invert y-axis so best is on top
        ax.invert_yaxis()
    
    plt.suptitle(f'Retrieval Metrics Breakdown (K={EVAL_K})', 
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = Path(OUTPUT_DIR) / "metrics_subplots.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")
    plt.close()

# ============================================================================
# PLOT 4: Ranking Table Visualization
# ============================================================================
def plot_ranking_table():
    """Create a clean ranking table visualization"""
    
    # Select top 10
    df_top10 = df.head(10).copy()
    df_top10['Rank'] = range(1, len(df_top10) + 1)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for _, row in df_top10.iterrows():
        table_data.append([
            row['Rank'],
            row['Config_Short'],
            f"{row['mrr']:.3f}",
            f"{row[f'hit_rate@{EVAL_K}']:.3f}",
            f"{row[f'precision@{EVAL_K}']:.3f}",
            f"{row[f'recall@{EVAL_K}']:.3f}",
            f"{row['source_acc']:.3f}"
        ])
    
    # Column headers
    columns = ['Rank', 'Configuration', 'mrr', f'Hit@{EVAL_K}', 
               f'Prec@{EVAL_K}', f'Rec@{EVAL_K}', 'Src Acc']
    
    table = ax.table(cellText=table_data, colLabels=columns,
                    cellLoc='center', loc='center',
                    colColours=['#2E86AB']*len(columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#2E86AB')
        cell.set_text_props(weight='bold', color='white', fontsize=12)
    
    # Style rows with alternating colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(columns)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#F0F0F0')
            else:
                cell.set_facecolor('white')
            
            # Highlight rank column
            if j == 0:
                cell.set_text_props(weight='bold', fontsize=12)
    
    plt.title(f'Top 10 Retrieval Configurations - Leaderboard (K={EVAL_K})', 
              fontsize=16, fontweight='bold', pad=20)
    
    output_path = Path(OUTPUT_DIR) / "ranking_table.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")
    plt.close()

# ============================================================================
# PLOT 5: Radar Chart for Top 5
# ============================================================================
def plot_radar_chart():
    """Radar chart comparing top 5 configurations"""
    
    df_top5 = df.head(5).copy()
    
    # Metrics for radar chart
    metrics = ['mrr', f'hit_rate@{EVAL_K}', f'precision@{EVAL_K}', 
               f'recall@{EVAL_K}', 'source_acc']
    
    # Number of variables
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    for idx, (_, row) in enumerate(df_top5.iterrows()):
        values = [row[m] for m in metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Config_Short'], 
                color=colors[idx], markersize=6)
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    # Fix axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace(f'@{EVAL_K}', f'\n@{EVAL_K}') for m in metrics], 
                       fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10, frameon=True)
    plt.title(f'Top 5 Configurations - Radar Comparison (K={EVAL_K})', 
              fontsize=14, fontweight='bold', pad=20)
    
    output_path = Path(OUTPUT_DIR) / "radar_chart_top5.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")
    plt.close()

# ============================================================================
# PLOT 6: Simple Bar Chart - mrr Only (for quick comparison)
# ============================================================================
def plot_mrr_simple():
    """Simple, clean mrr comparison"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create color gradient based on mrr scores
    colors = plt.cm.RdYlGn(df['mrr'] / df['mrr'].max())
    
    bars = ax.barh(df['Config_Short'], df['mrr'], color=colors, 
                   edgecolor='black', linewidth=1.5, alpha=0.9)
    
    # Add value labels
    for bar, val in zip(bars, df['mrr']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{val:.3f}', 
               va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('mrr Score', fontsize=14, fontweight='bold')
    ax.set_title(f'Mean Reciprocal Rank (mrr) Comparison (K={EVAL_K})', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(0, 1.1)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()
    
    plt.tight_layout()
    output_path = Path(OUTPUT_DIR) / "mrr_comparison_simple.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("🎨 Generating presentation-quality plots...\n")
    
    plot_top_configs_grouped()
    plot_heatmap_enhanced()
    plot_metric_subplots()
    plot_ranking_table()
    plot_radar_chart()
    plot_mrr_simple()
    
    print("\n✨ All plots generated successfully!")
    print(f"📁 Check output directory: {OUTPUT_DIR}")