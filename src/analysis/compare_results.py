"""
Compare the performance of baseline Naive Bayes classifier with the enhanced SVM classifier
for bug report classification across multiple datasets.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Define datasets
DATASETS = ['pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe']

# Define metrics to compare
METRICS = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']

def get_result_directories():
    """Determine the baseline and enhanced results directories."""
    # Check environment variables first (set by main.py)
    baseline_dir = os.environ.get('BASELINE_RESULTS_DIR', 'baseline_results')
    enhanced_dir = os.environ.get('ENHANCED_RESULTS_DIR', 'results')
    analysis_dir = os.environ.get('ANALYSIS_OUTPUT_DIR', os.path.join(enhanced_dir, 'analysis'))
    
    # Check if we need to convert to absolute paths
    if not os.path.isabs(baseline_dir):
        # Try multiple possible locations
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        
        possible_baseline_dirs = [
            os.path.join(project_root, baseline_dir),
            baseline_dir,
            os.path.join(os.path.dirname(project_root), baseline_dir),
            os.path.join(project_root, "baseline_results")
        ]
        
        for dir_path in possible_baseline_dirs:
            if os.path.exists(dir_path):
                baseline_dir = dir_path
                break
    
    if not os.path.isabs(enhanced_dir):
        # Try multiple possible locations
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        
        possible_enhanced_dirs = [
            os.path.join(project_root, enhanced_dir),
            enhanced_dir,
            os.path.join(os.path.dirname(project_root), enhanced_dir),
            os.path.join(project_root, "results")
        ]
        
        for dir_path in possible_enhanced_dirs:
            if os.path.exists(dir_path):
                enhanced_dir = dir_path
                break
    
    # Ensure the analysis directory exists
    if not os.path.isabs(analysis_dir):
        analysis_dir = os.path.join(os.path.dirname(enhanced_dir), 'analysis')
    
    os.makedirs(analysis_dir, exist_ok=True)
    
    return baseline_dir, enhanced_dir, analysis_dir

def load_results(base_dir, suffix, datasets):
    """Load results from CSV files for all datasets."""
    results = {}
    for dataset in datasets:
        file_path = os.path.join(base_dir, f"{dataset}_{suffix}.csv")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                # Take the first row if there are multiple rows
                results[dataset] = df.iloc[0].to_dict()
                logger.info(f"Loaded {dataset}_{suffix} results")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        else:
            logger.warning(f"Warning: File not found - {file_path}")
    return results

def compute_improvements(baseline_results, enhanced_results, datasets, metrics):
    """Compute the improvement percentages between baseline and enhanced results."""
    improvements = {}
    
    for dataset in datasets:
        if dataset in baseline_results and dataset in enhanced_results:
            baseline = baseline_results[dataset]
            enhanced = enhanced_results[dataset]
            
            improvements[dataset] = {}
            for metric in metrics:
                if metric in baseline and metric in enhanced:
                    baseline_value = baseline[metric]
                    enhanced_value = enhanced[metric]
                    absolute_diff = enhanced_value - baseline_value
                    percentage_diff = (absolute_diff / baseline_value) * 100 if baseline_value != 0 else 0
                    
                    improvements[dataset][metric] = {
                        'baseline': baseline_value,
                        'enhanced': enhanced_value,
                        'absolute_diff': absolute_diff,
                        'percentage_diff': percentage_diff
                    }
    
    return improvements

def create_summary_dataframe(improvements, datasets, metrics):
    """Create a summary DataFrame from the improvements data."""
    data = []
    
    for dataset in datasets:
        if dataset in improvements:
            for metric in metrics:
                if metric in improvements[dataset]:
                    imp = improvements[dataset][metric]
                    
                    data.append({
                        'Dataset': dataset,
                        'Metric': metric,
                        'Baseline (NB)': imp['baseline'],
                        'Enhanced (SVM)': imp['enhanced'],
                        'Absolute Diff': imp['absolute_diff'],
                        'Percentage Diff (%)': imp['percentage_diff']
                    })
    
    return pd.DataFrame(data)

def plot_comparison(summary_df, output_dir):
    """Create visualization plots comparing baseline and enhanced models."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seaborn style
    sns.set(style="whitegrid")
    
    # 1. Bar plot comparing baseline vs enhanced for each metric across datasets
    for metric in METRICS:
        metric_df = summary_df[summary_df['Metric'] == metric]
        
        plt.figure(figsize=(12, 6))
        
        # Prepare data for side-by-side bars
        data_to_plot = metric_df.melt(
            id_vars=['Dataset'], 
            value_vars=['Baseline (NB)', 'Enhanced (SVM)'],
            var_name='Model', 
            value_name='Value'
        )
        
        # Plot bars
        ax = sns.barplot(x='Dataset', y='Value', hue='Model', data=data_to_plot)
        
        # Add value labels on top of bars
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha="center")
        
        # Set plot title and labels
        plt.title(f'Comparison of {metric} Between Baseline (NB) and Enhanced (SVM)', fontsize=14)
        plt.xlabel('Dataset', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.ylim(0, max(data_to_plot['Value']) * 1.15)  # Add some space for labels
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_comparison.png'))
        plt.close()
    
    # 2. Heatmap of percentage improvements across all metrics and datasets
    pivot_df = summary_df.pivot(index='Dataset', columns='Metric', values='Percentage Diff (%)')
    
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', center=0, fmt='.1f', linewidths=.5)
    plt.title('Percentage Improvement of Enhanced (SVM) over Baseline (NB)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'percentage_improvement_heatmap.png'))
    plt.close()
    
    # 3. Radar chart for each dataset to compare across all metrics
    # Function to create radar chart
    def create_radar_chart(dataset, baseline_values, enhanced_values, metrics):
        # Convert to numpy arrays for easier calculation
        baseline_values = np.array(baseline_values)
        enhanced_values = np.array(enhanced_values)
        
        # Calculate max for normalization
        maxs = [max(b, e) for b, e in zip(baseline_values, enhanced_values)]
        
        # Normalize values between 0 and 1 for better visualization
        norm_baseline = [b/m if m != 0 else 0 for b, m in zip(baseline_values, maxs)]
        norm_enhanced = [e/m if m != 0 else 0 for e, m in zip(enhanced_values, maxs)]
        
        # Set up radar chart
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the circle
        
        norm_baseline += norm_baseline[:1]  # Close the baseline polygon
        norm_enhanced += norm_enhanced[:1]  # Close the enhanced polygon
        
        metrics_labels = metrics + [metrics[0]]  # Close the labels
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        
        # Plot baseline
        ax.plot(angles, norm_baseline, 'o-', linewidth=2, label='Baseline (NB)')
        ax.fill(angles, norm_baseline, alpha=0.1)
        
        # Plot enhanced
        ax.plot(angles, norm_enhanced, 'o-', linewidth=2, label='Enhanced (SVM)')
        ax.fill(angles, norm_enhanced, alpha=0.1)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        
        # Add actual values as annotations
        for i, (m, nb, mb) in enumerate(zip(metrics, baseline_values, enhanced_values)):
            angle = angles[i]
            ax.text(angle, norm_baseline[i] + 0.05, f"{nb:.3f}", ha='center', va='center')
            ax.text(angle, norm_enhanced[i] + 0.05, f"{mb:.3f}", ha='center', va='center', color='red')
        
        plt.title(f'Performance Comparison for {dataset} Dataset', fontsize=14)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{dataset}_radar.png'))
        plt.close()
    
    # Create radar charts for each dataset
    for dataset in DATASETS:
        dataset_df = summary_df[summary_df['Dataset'] == dataset]
        if not dataset_df.empty:
            baseline_values = []
            enhanced_values = []
            
            # Maintain consistent order of metrics
            for metric in METRICS:
                metric_row = dataset_df[dataset_df['Metric'] == metric]
                if not metric_row.empty:
                    baseline_values.append(metric_row['Baseline (NB)'].values[0])
                    enhanced_values.append(metric_row['Enhanced (SVM)'].values[0])
            
            if baseline_values and enhanced_values:
                create_radar_chart(dataset, baseline_values, enhanced_values, METRICS)
    
    # 4. Grouped bar chart showing F1 and AUC across all datasets
    key_metrics = ['F1', 'AUC']
    key_metrics_df = summary_df[summary_df['Metric'].isin(key_metrics)]
    
    plt.figure(figsize=(15, 8))
    
    # Prepare data
    data_to_plot = key_metrics_df.melt(
        id_vars=['Dataset', 'Metric'], 
        value_vars=['Baseline (NB)', 'Enhanced (SVM)'],
        var_name='Model', 
        value_name='Value'
    )
    
    # Plot grouped bars
    g = sns.catplot(
        x='Dataset', y='Value', hue='Model', col='Metric',
        data=data_to_plot, kind='bar', height=6, aspect=1.2
    )
    
    # Customize plots
    g.set_axis_labels("Dataset", "Value")
    g.set_titles("{col_name}")
    g.set_xticklabels(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'key_metrics_comparison.png'))
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def generate_report(summary_df, improvements, output_file):
    """Generate a detailed report of the comparison results."""
    with open(output_file, 'w') as f:
        # Write report header
        f.write("# Bug Report Classification: Baseline vs. Enhanced Model Comparison\n\n")
        f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall summary
        f.write("## Overall Performance Comparison\n\n")
        
        # Create comparison table
        table_data = []
        for dataset in DATASETS:
            if dataset in improvements:
                for model_type, model_name in [("baseline", "NB"), ("enhanced", "SVM")]:
                    row = [dataset, model_name]
                    for metric in METRICS:
                        if metric in improvements[dataset]:
                            row.append(improvements[dataset][metric][model_type])
                    table_data.append(row)
        
        # Write table to report
        f.write(tabulate(
            table_data, 
            headers=["Dataset", "Model"] + METRICS,
            tablefmt="pipe",
            floatfmt=".4f"
        ))
        f.write("\n\n")
        
        # Detailed analysis by metric
        f.write("## Detailed Analysis by Metric\n\n")
        
        for metric in METRICS:
            f.write(f"### {metric}\n\n")
            
            metric_data = []
            for dataset in DATASETS:
                if dataset in improvements and metric in improvements[dataset]:
                    imp = improvements[dataset][metric]
                    metric_data.append([
                        dataset,
                        imp['baseline'],
                        imp['enhanced'],
                        imp['absolute_diff'],
                        f"{imp['percentage_diff']:.2f}%"
                    ])
            
            f.write(tabulate(
                metric_data,
                headers=["Dataset", "Baseline (NB)", "Enhanced (SVM)", "Absolute Difference", "Percentage Difference"],
                tablefmt="pipe",
                floatfmt=".4f"
            ))
            f.write("\n\n")
        
        # Key observations
        f.write("## Key Observations\n\n")
        
        # Calculate average improvements by metric
        avg_improvements = {}
        for metric in METRICS:
            values = [improvements[dataset][metric]['percentage_diff'] 
                    for dataset in DATASETS 
                    if dataset in improvements and metric in improvements[dataset]]
            if values:
                avg_improvements[metric] = sum(values) / len(values)
        
        # Sort metrics by average improvement
        sorted_metrics = sorted(avg_improvements.items(), key=lambda x: x[1], reverse=True)
        
        # Write observations about metrics with the most improvement
        f.write(f"1. **Most Improved Metric**: {sorted_metrics[0][0]} with an average improvement of {sorted_metrics[0][1]:.2f}%\n")
        f.write(f"2. **Second Most Improved**: {sorted_metrics[1][0]} with an average improvement of {sorted_metrics[1][1]:.2f}%\n\n")
        
        # Identify datasets with most improvement in F1 score
        f1_improvements = [(dataset, improvements[dataset]['F1']['percentage_diff']) 
                         for dataset in DATASETS 
                         if dataset in improvements and 'F1' in improvements[dataset]]
        
        sorted_f1 = sorted(f1_improvements, key=lambda x: x[1], reverse=True)
        
        f.write("### Dataset-Specific Observations\n\n")
        
        for dataset, improvement in sorted_f1:
            f.write(f"- **{dataset}**: F1 score improved by {improvement:.2f}%\n")
        
        f.write("\n")
        
        # Overall assessment
        f.write("## Overall Assessment\n\n")
        
        # Calculate number of metrics that improved overall
        improved_metrics = [metric for metric, avg_imp in avg_improvements.items() if avg_imp > 0]
        
        f.write(f"The enhanced SVM model shows improvements in {len(improved_metrics)} out of {len(METRICS)} metrics on average across all datasets.\n\n")
        
        # Add some specific observations about the most notable improvements
        if improved_metrics:
            f.write("Notable improvements:\n\n")
            for metric in improved_metrics:
                f.write(f"- **{metric}**: Average improvement of {avg_improvements[metric]:.2f}%\n")
        
        f.write("\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        
        # Make an overall conclusion based on the F1 and AUC improvements
        if 'F1' in avg_improvements and 'AUC' in avg_improvements:
            if avg_improvements['F1'] > 0 and avg_improvements['AUC'] > 0:
                f.write("The enhanced SVM classifier demonstrates significant improvements over the baseline Naive Bayes model, particularly in F1 score and AUC, indicating better overall classification performance and discriminative ability.\n")
            elif avg_improvements['F1'] > 0:
                f.write("The enhanced SVM classifier improves F1 score over the baseline Naive Bayes model, indicating better balance between precision and recall.\n")
            elif avg_improvements['AUC'] > 0:
                f.write("The enhanced SVM classifier improves AUC over the baseline Naive Bayes model, indicating better discriminative ability.\n")
            else:
                f.write("The enhanced SVM classifier shows mixed results compared to the baseline Naive Bayes model.\n")
        
        # Add note about visualizations
        f.write("\n")
        f.write("*Note: Refer to the generated visualizations for graphical representation of these comparisons.*\n")
    
    print(f"Report generated and saved to {output_file}")

def main():
    """Main function to run the comparison analysis."""
    # Get results directories
    baseline_dir, enhanced_dir, analysis_dir = get_result_directories()
    
    logger.info(f"Loading baseline NB results from: {baseline_dir}")
    baseline_results = load_results(baseline_dir, "NB", DATASETS)
    
    logger.info(f"Loading enhanced SVM results from: {enhanced_dir}")
    enhanced_results = load_results(enhanced_dir, "SVM", DATASETS)
    
    if not baseline_results or not enhanced_results:
        logger.error("Could not load required results. Exiting.")
        return
    
    logger.info("Computing improvements...")
    improvements = compute_improvements(baseline_results, enhanced_results, DATASETS, METRICS)
    
    logger.info("Creating summary dataframe...")
    summary_df = create_summary_dataframe(improvements, DATASETS, METRICS)
    
    # Save summary DataFrame as CSV
    summary_file = os.path.join(analysis_dir, "comparison_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Summary saved to {summary_file}")
    
    # Plot comparison visualizations
    logger.info("Creating visualizations...")
    plot_comparison(summary_df, analysis_dir)
    
    # Generate detailed report
    report_file = os.path.join(analysis_dir, "comparison_report.md")
    logger.info("Generating detailed report...")
    generate_report(summary_df, improvements, report_file)
    
    logger.info(f"Comparison analysis completed. Results saved to {analysis_dir}")
    
    # Print a summary table to console
    logger.info("\nSummary of Percentage Improvements:")
    pivot_table = summary_df.pivot(index='Dataset', columns='Metric', values='Percentage Diff (%)')
    logger.info(tabulate(pivot_table, headers='keys', tablefmt='grid', floatfmt='.2f'))

if __name__ == "__main__":
    main() 