import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

plt.rcParams['font.size'] = 11

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_comparison_bars(comparison_data):
    """Single clean comparison chart - your main slide"""
    comp = comparison_data['comparison']
    
    metrics = ['Precision@4', 'Recall@4', 'MRR', 'Hit Rate@4', 
               'Source Accuracy', 'Context Completeness']
    
    rag_only = [comp['precision@4']['rag_only'], 
                comp['recall@4']['rag_only'],
                comp['mrr']['rag_only'],
                comp['hit_rate@4']['rag_only'],
                comp['source_accuracy']['rag_only'],
                comp['context_completeness']['rag_only']]
    
    rag_kg = [comp['precision@4']['parallel'],
              comp['recall@4']['parallel'],
              comp['mrr']['parallel'],
              comp['hit_rate@4']['parallel'],
              comp['source_accuracy']['parallel'],
              comp['context_completeness']['parallel']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, rag_only, width, label='RAG Only', color='#3498db', alpha=0.85)
    ax.bar(x + width/2, rag_kg, width, label='RAG + KG', color='#e74c3c', alpha=0.85)
    
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('System Performance Comparison', fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30, ha='right')
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=12, loc='upper right')
    
    # White background, no grid
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    return fig

def plot_improvement_bars(comparison_data):
    """Show % improvements - your impact slide"""
    comp = comparison_data['comparison']
    
    metrics = ['Precision@4', 'Recall@4', 'MRR', 'Hit Rate@4', 
               'Source Accuracy', 'Context\nCompleteness']
    
    improvements = [
        comp['precision@4']['improvement_pct'],
        comp['recall@4']['improvement_pct'],
        comp['mrr']['improvement_pct'],
        comp['hit_rate@4']['improvement_pct'],
        comp['source_accuracy']['improvement_pct'],
        comp['context_completeness']['improvement_pct']
    ]
    
    colors = ['#27ae60' if x > 0 else '#c0392b' for x in improvements]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(metrics, improvements, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Improvement (%)', fontsize=13, fontweight='bold')
    ax.set_title('RAG+KG Performance Change', fontsize=15, fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
    
    # White background, no grid
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Extend x-axis limits to give more room for labels
    ax.set_xlim(-18, 25)
    
    # Add value labels
    for bar, val in zip(bars, improvements):
        width = bar.get_width()
        # Adjust position to avoid overlap with y-axis labels
        if width > 0:
            x_pos = width + 1.5
            ha = 'left'
        else:
            x_pos = width - 2.0  # More spacing for negative values
            ha = 'right'
        
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, 
                f'{val:+.1f}%',
                ha=ha, va='center', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Simple RAG evaluation visualizer')
    parser.add_argument('--input_dir', type=str, default='./evaluation_results',
                        help='Directory containing JSON files (default: ./evaluation_results)')
    parser.add_argument('--comparison', type=str, default=None,
                        help='Comparison JSON filename (default: auto-detect latest)')
    parser.add_argument('--output_dir', type=str, default='./figures',
                        help='Output directory for plots (default: ./figures)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    # Auto-detect comparison file if not specified
    if args.comparison is None:
        comparison_files = list(input_dir.glob('comparison*.json'))
        if not comparison_files:
            print(f"Error: No comparison*.json files found in {input_dir}")
            exit(1)
        # Get the most recent one
        input_path = max(comparison_files, key=lambda p: p.stat().st_mtime)
        print(f"Auto-detected: {input_path.name}")
    else:
        input_path = input_dir / args.comparison
    
    if not input_path.exists():
        print(f"Error: Could not find {input_path}")
        print(f"Looking in: {input_dir.absolute()}")
        exit(1)

    print(f"Loading data from: {input_path}")
    comparison_data = load_json(input_path)
    
    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate 2 key plots
    print("Generating visualizations...")
    
    print("  [1/2] Comparison chart...")
    fig1 = plot_comparison_bars(comparison_data)
    fig1.savefig(output_dir / 'comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  [2/2] Improvement chart...")
    fig2 = plot_improvement_bars(comparison_data)
    fig2.savefig(output_dir / 'improvement.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Done! Saved to {output_dir}/")
    print("\nFiles created:")
    print(f"  - comparison.png  (side-by-side scores)")
    print(f"  - improvement.png (% improvements)")
    
    # Print key numbers
    print("\n" + "="*50)
    print("KEY TAKEAWAYS:")
    print("="*50)
    comp = comparison_data['comparison']
    print(f"✓ Context Completeness: {comp['context_completeness']['improvement_pct']:+.1f}%")
    print(f"✓ MRR (ranking):        {comp['mrr']['improvement_pct']:+.1f}%")
    print(f"✓ Precision:            {comp['precision@4']['improvement_pct']:+.1f}%")
    print(f"✗ Recall:               {comp['recall@4']['improvement_pct']:+.1f}%")

if __name__ == "__main__":
    main()