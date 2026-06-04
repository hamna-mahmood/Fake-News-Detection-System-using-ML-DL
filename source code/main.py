import warnings
import os
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns           
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
warnings.filterwarnings('ignore')

def print_detailed_metrics(y_true, y_pred, model_name):
    """Print all 4 metrics for a model"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\n {model_name} METRICS:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def print_model_comparison_table(models_results):
    """Comparison table of all metrics"""
    print("\n" + "="*100)
    print("COMPLETE MODEL COMPARISON")
    print("="*100)
    print(f"{'Model':<18} {'Acc':<7} {'Prec':<7} {'Rec':<7} {'F1':<7} {'Best Metric'}")
    print("-"*100)
    
    for model_name, result in models_results.items():
        if result['success']:
            metrics = result['metrics']
            best_metric = max(['accuracy', 'precision', 'recall', 'f1'], 
                            key=lambda x: metrics[x])
            print(f"{model_name:<18} {metrics['accuracy']:<7.4f} "
                  f"{metrics['precision']:<7.4f} {metrics['recall']:<7.4f} "
                  f"{metrics['f1']:<7.4f} {best_metric}")
    
    # Overall best model by F1-score
    best_f1_model = max(
        [(name, result['metrics']['f1']) for name, result in models_results.items() if result['success']],
        key=lambda x: x[1]
    )
    print("-"*100)
    print(f"BEST F1-SCORE: {best_f1_model[0]} (F1: {best_f1_model[1]:.4f})")
    print("="*100)

def plot_all_metrics_comparison(models_results):
    """Create 4 PROFESSIONAL graphs: Accuracy, Precision, Recall, F1"""
    
    # Extract all successful models' metrics
    successful_models = {}
    for name, result in models_results.items():
        if result['success']:
            successful_models[name] = result['metrics']
    
    if not successful_models:
        print("No successful models to plot!")
        return
    
    model_names = list(successful_models.keys())
    metrics_data = {
        'Accuracy': [successful_models[name]['accuracy'] for name in model_names],
        'Precision': [successful_models[name]['precision'] for name in model_names],
        'Recall': [successful_models[name]['recall'] for name in model_names],
        'F1-Score': [successful_models[name]['f1'] for name in model_names]
    }
    
    # 🎨 COLORS
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # CREATE 2x2 DASHBOARD
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('COMPLETE MODEL PERFORMANCE COMPARISON\nFake News Detection Pipeline', 
                fontsize=28, fontweight='bold', y=0.98, color='#2C3E50')
    
    # Metric names and positions
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    positions = [(0,0), (0,1), (1,0), (1,1)]
    
    best_scores = {}
    best_models = {}
    
    for i, (metric, (row, col)) in enumerate(zip(metrics, positions)):
        ax = axes[row, col]
        
        # Sort bars by score (descending)
        sorted_idx = np.argsort(metrics_data[metric])[::-1]
        sorted_names = [model_names[j] for j in sorted_idx]
        sorted_scores = [metrics_data[metric][j] for j in sorted_idx]
        
        # Create bars
        bars = ax.barh(range(len(sorted_names)), sorted_scores,
                      color=[colors[j % len(colors)] for j in sorted_idx],
                      edgecolor='black', linewidth=1.8, height=0.65, alpha=0.85)
        
        # HIGHLIGHT BEST MODEL
        best_idx = 0
        bars[best_idx].set_color('#00A676')
        bars[best_idx].set_edgecolor('#00855A')
        bars[best_idx].set_linewidth(3)
        best_scores[metric] = sorted_scores[0]
        best_models[metric] = sorted_names[0]
        
        # LABELS 
        for j, score in enumerate(sorted_scores):
            # Main score label
            ax.text(score + 0.008, j, f'{score:.4f}', 
                   ha='left', va='center', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none'))
            
            # Percentage label
            ax.text(score + 0.035, j, f'({score*100:.1f}%)', 
                   ha='left', va='center', fontsize=11, color='#666')
        
        # Axis formatting
        ax.set_xlim(0, 1.06)
        ax.set_ylim(-0.5, len(sorted_names)-0.5)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names, fontsize=12, fontweight='medium')
        
        # X-axis percentages
        ax.set_xticks([0.0, 0.85, 0.90, 0.95, 1.00])
        ax.set_xticklabels(['0%', '85%', '90%', '95%', '100%'], fontsize=11)
        
        # Title & labels
        ax.set_title(f'{metric}', fontsize=20, fontweight='bold', pad=20, color='#34495E')
        ax.set_xlabel('Score', fontsize=13, fontweight='bold')
        
        # Clean design
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.grid(axis='x', alpha=0.25, linestyle='-', linewidth=0.8)
    
    # SUMMARY TABLE
    champion_text = f"""CHAMPIONS BY METRIC:
- Accuracy:  {best_scores['Accuracy']:.4f} ({best_models['Accuracy']})
- Precision: {best_scores['Precision']:.4f} ({best_models['Precision']})
- Recall:    {best_scores['Recall']:.4f} ({best_models['Recall']})
- F1-Score:  {best_scores['F1-Score']:.4f} ({best_models['F1-Score']})"""
    
    plt.figtext(0.02, 0.02, champion_text,
               fontsize=15, fontweight='bold', color='#00A676',
               bbox=dict(boxstyle='round,pad=1.0', facecolor='white', alpha=0.95, 
                        edgecolor='#00A676', linewidth=2),
               verticalalignment='bottom')
    
    # SAVE
    plt.tight_layout()
    plt.savefig('model_comparison_dashboard.png', dpi=400, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("\n MASTER DASHBOARD saved as 'model_comparison_dashboard.png'")
    print(" 4 Professional graphs generated! ✨")
    
    plt.show()

def main():
    print("Fake News Detection Pipeline - FULL METRICS")
    print("=" * 60)
    
    models_results = {}
    
    # 1. ML MODELS
    print("\n 1. ML MODELS (TF-IDF)")
    print("-" * 40)
    try:
        from preprocess_ml import preprocess_ml_data
        X_train_ml, X_test_ml, y_train_ml, y_test_ml, _ = preprocess_ml_data()
        print(f"ML Data loaded: Train={X_train_ml.shape[0]}, Test={X_test_ml.shape[0]}")
        
        # Logistic Regression
        print("\n - Logistic Regression...")
        from logistic_regression import train_logistic_regression
        lr_model = train_logistic_regression(X_train_ml, X_test_ml, y_train_ml, y_test_ml)
        y_pred_lr = lr_model.predict(X_test_ml)
        lr_metrics = print_detailed_metrics(y_test_ml, y_pred_lr, "Logistic Regression")
        models_results['Logistic Regression'] = {'metrics': lr_metrics, 'success': True}
        
        # Decision Tree
        print("\n - Decision Tree...")
        from decision_tree import train_decision_tree
        dt_model = train_decision_tree(X_train_ml, X_test_ml, y_train_ml, y_test_ml)
        y_pred_dt = dt_model.predict(X_test_ml)
        dt_metrics = print_detailed_metrics(y_test_ml, y_pred_dt, "Decision Tree")
        models_results['Decision Tree'] = {'metrics': dt_metrics, 'success': True}
        
        # Random Forest
        print("\n - Random Forest...")
        from random_forest import train_random_forest
        rf_model = train_random_forest(X_train_ml, X_test_ml, y_train_ml, y_test_ml)
        y_pred_rf = rf_model.predict(X_test_ml)
        rf_metrics = print_detailed_metrics(y_test_ml, y_pred_rf, "Random Forest")
        models_results['Random Forest'] = {'metrics': rf_metrics, 'success': True}
        
    except Exception as e:
        print(f" ML Error: {e}")
    
    # 2. DL MODELS
    print("\n 2. DEEP LEARNING MODELS")
    print("-" * 40)
    try:
        from preprocess_dl import get_dl_data
        X_train_pad, X_test_pad, y_train, y_test, tokenizer = get_dl_data()
        
        # CNN
        print("\n - CNN...")
        from cnn import run_cnn
        cnn_model = run_cnn()
        y_pred_cnn = (cnn_model.predict(X_test_pad, verbose=0) > 0.5).astype("int32")
        cnn_metrics = print_detailed_metrics(y_test, y_pred_cnn, "CNN")
        models_results['CNN'] = {'metrics': cnn_metrics, 'success': True}
        
        # LSTM
        print("\n - LSTM...")
        from lstm import run_lstm
        lstm_model = run_lstm()
        y_pred_lstm = (lstm_model.predict(X_test_pad, verbose=0) > 0.5).astype("int32")
        lstm_metrics = print_detailed_metrics(y_test, y_pred_lstm, "LSTM")
        models_results['LSTM'] = {'metrics': lstm_metrics, 'success': True}
        
    except Exception as e:
        print(f" DL Error: {e}")
    
    # FINAL COMPARISON TABLE
    print_model_comparison_table(models_results)
    
    # 4 GRAPH DASHBOARD
    print("\n Creating 4-metric comparison dashboard...")
    plot_all_metrics_comparison(models_results)
    
    print("\n FULL PIPELINE COMPLETE!")
    print(" Professional 4-graph dashboard generated!")

if __name__ == "__main__":
    main()