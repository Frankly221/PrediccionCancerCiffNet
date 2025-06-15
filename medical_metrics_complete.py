import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, f1_score, precision_recall_curve,
    balanced_accuracy_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import pandas as pd
from datetime import datetime
import os

class MedicalMetricsComplete:
    """Métricas médicas completas con visualizaciones para artículo científico"""
    
    def __init__(self, class_names):
        self.class_names = class_names
        self.melanoma_idx = self._find_melanoma_index()
        self.malignant_classes = ['mel', 'bcc', 'akiec']  # Clases malignas/pre-malignas
        
        # Colores para visualizaciones
        self.colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
        
        # Crear directorio para gráficos
        self.figures_dir = "medical_analysis_figures"
        os.makedirs(self.figures_dir, exist_ok=True)
        
    def _find_melanoma_index(self):
        for i, name in enumerate(self.class_names):
            if 'mel' in name.lower():
                return i
        return None
    
    def compute_comprehensive_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Métricas médicas completas"""
        
        # 1. MÉTRICAS BÁSICAS
        basic_metrics = {
            'accuracy': np.mean(y_true == y_pred) * 100,
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred) * 100,
            'total_samples': len(y_true)
        }
        
        # 2. MÉTRICAS POR CLASE
        class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)
            
            tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
            tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
            fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
            
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[class_name] = {
                'recall': recall * 100,      # Sensibilidad
                'precision': precision * 100,
                'specificity': specificity * 100,
                'f1_score': f1 * 100,
                'support': np.sum(y_true_binary),
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
            }
        
        # 3. MÉTRICAS CRÍTICAS DE MELANOMA
        melanoma_metrics = {}
        if self.melanoma_idx is not None:
            mel_class = self.class_names[self.melanoma_idx]
            melanoma_metrics = {
                'melanoma_recall': class_metrics[mel_class]['recall'],
                'melanoma_precision': class_metrics[mel_class]['precision'],
                'melanoma_f1': class_metrics[mel_class]['f1_score'],
                'melanoma_specificity': class_metrics[mel_class]['specificity'],
                'melanoma_support': class_metrics[mel_class]['support']
            }
        
        # 4. MÉTRICAS MALIGNO vs BENIGNO
        malignant_vs_benign = self._compute_malignant_benign_metrics(y_true, y_pred)
        
        # 5. AUC y curvas ROC
        auc_metrics = {}
        roc_data = {}
        if y_pred_proba is not None:
            auc_metrics, roc_data = self._compute_auc_metrics(y_true, y_pred_proba)
        
        # 6. Matriz de confusión
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        return {
            'basic': basic_metrics,
            'by_class': class_metrics,
            'melanoma': melanoma_metrics,
            'malignant_benign': malignant_vs_benign,
            'auc': auc_metrics,
            'roc_data': roc_data,
            'confusion_matrix': conf_matrix,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _compute_malignant_benign_metrics(self, y_true, y_pred):
        """Métricas maligno vs benigno"""
        y_true_binary = np.array([1 if self.class_names[i].lower() in self.malignant_classes else 0 
                                 for i in y_true])
        y_pred_binary = np.array([1 if self.class_names[i].lower() in self.malignant_classes else 0 
                                 for i in y_pred])
        
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        return {
            'malignant_sensitivity': sensitivity * 100,
            'malignant_specificity': specificity * 100,
            'malignant_precision': precision * 100,
            'malignant_f1': f1 * 100,
            'malignant_support': np.sum(y_true_binary),
            'benign_support': np.sum(1 - y_true_binary)
        }
    
    def _compute_auc_metrics(self, y_true, y_pred_proba):
        """Compute AUC metrics and ROC data"""
        auc_metrics = {}
        roc_data = {}
        
        try:
            # AUC multiclase
            auc_macro = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
            auc_weighted = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            
            auc_metrics = {
                'auc_macro': auc_macro,
                'auc_weighted': auc_weighted
            }
            
            # ROC por clase (One-vs-Rest)
            y_true_binarized = label_binarize(y_true, classes=range(len(self.class_names)))
            
            for i, class_name in enumerate(self.class_names):
                if len(np.unique(y_true_binarized[:, i])) > 1:
                    fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_pred_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    
                    auc_metrics[f'auc_{class_name}'] = roc_auc
                    roc_data[class_name] = {
                        'fpr': fpr,
                        'tpr': tpr,
                        'auc': roc_auc
                    }
                    
        except Exception as e:
            print(f"Error calculando AUC: {e}")
            
        return auc_metrics, roc_data
    
    def create_comprehensive_visualizations(self, metrics, save_figs=True):
        """Crear todas las visualizaciones para artículo"""
        
        # 1. Matriz de Confusión
        self._plot_confusion_matrix(metrics, save_figs)
        
        # 2. Curvas ROC
        if metrics['roc_data']:
            self._plot_roc_curves(metrics, save_figs)
        
        # 3. Métricas por clase
        self._plot_class_metrics(metrics, save_figs)
        
        # 4. Análisis de melanoma
        self._plot_melanoma_analysis(metrics, save_figs)
        
        # 5. Distribución de clases
        self._plot_class_distribution(metrics, save_figs)
        
        # 6. Resumen ejecutivo
        self._plot_executive_summary(metrics, save_figs)
        
        print(f"✅ Visualizaciones guardadas en: {self.figures_dir}/")
    
    def _plot_confusion_matrix(self, metrics, save_figs):
        """Matriz de confusión normalizada"""
        plt.figure(figsize=(12, 10))
        
        # Normalizar por filas (verdaderos)
        cm = metrics['confusion_matrix']
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Crear heatmap
        sns.heatmap(cm_normalized, 
                   annot=True, 
                   fmt='.2f',
                   cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Recall (Normalized)'})
        
        plt.title('Confusion Matrix - Normalized by True Class\n(Skin Cancer Classification)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Class', fontsize=14)
        plt.ylabel('True Class', fontsize=14)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # Añadir números absolutos como texto
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                plt.text(j+0.5, i+0.7, f'({cm[i,j]})', 
                        ha='center', va='center', fontsize=10, color='red')
        
        plt.tight_layout()
        if save_figs:
            plt.savefig(f'{self.figures_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{self.figures_dir}/confusion_matrix.pdf', bbox_inches='tight')
        plt.show()
    
    def _plot_roc_curves(self, metrics, save_figs):
        """Curvas ROC para todas las clases"""
        plt.figure(figsize=(14, 10))
        
        roc_data = metrics['roc_data']
        
        # Plot ROC curve para cada clase
        for i, (class_name, data) in enumerate(roc_data.items()):
            plt.plot(data['fpr'], data['tpr'], 
                    color=self.colors[i], 
                    lw=2.5,
                    label=f'{class_name.upper()} (AUC = {data["auc"]:.3f})')
        
        # Línea diagonal (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.8, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
        plt.title('ROC Curves - Skin Cancer Classification\nOne-vs-Rest Approach', 
                 fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Destacar melanoma si existe
        if self.melanoma_idx is not None:
            mel_class = self.class_names[self.melanoma_idx]
            if mel_class in roc_data:
                mel_data = roc_data[mel_class]
                plt.plot(mel_data['fpr'], mel_data['tpr'], 
                        color='red', lw=4, alpha=0.7,
                        label=f'{mel_class.upper()} - HIGHLIGHTED')
        
        plt.tight_layout()
        if save_figs:
            plt.savefig(f'{self.figures_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{self.figures_dir}/roc_curves.pdf', bbox_inches='tight')
        plt.show()
    
    def _plot_class_metrics(self, metrics, save_figs):
        """Métricas por clase - bar plot"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        classes = list(metrics['by_class'].keys())
        
        # Recall
        recalls = [metrics['by_class'][c]['recall'] for c in classes]
        axes[0,0].bar(classes, recalls, color=self.colors[:len(classes)], alpha=0.8)
        axes[0,0].set_title('Recall (Sensitivity) by Class', fontweight='bold')
        axes[0,0].set_ylabel('Recall (%)')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # Precision
        precisions = [metrics['by_class'][c]['precision'] for c in classes]
        axes[0,1].bar(classes, precisions, color=self.colors[:len(classes)], alpha=0.8)
        axes[0,1].set_title('Precision by Class', fontweight='bold')
        axes[0,1].set_ylabel('Precision (%)')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        # F1-Score
        f1_scores = [metrics['by_class'][c]['f1_score'] for c in classes]
        axes[1,0].bar(classes, f1_scores, color=self.colors[:len(classes)], alpha=0.8)
        axes[1,0].set_title('F1-Score by Class', fontweight='bold')
        axes[1,0].set_ylabel('F1-Score (%)')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].grid(True, alpha=0.3)
        
        # Support (número de muestras)
        supports = [metrics['by_class'][c]['support'] for c in classes]
        axes[1,1].bar(classes, supports, color=self.colors[:len(classes)], alpha=0.8)
        axes[1,1].set_title('Sample Support by Class', fontweight='bold')
        axes[1,1].set_ylabel('Number of Samples')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].grid(True, alpha=0.3)
        
        # Destacar melanoma
        if self.melanoma_idx is not None:
            mel_class = self.class_names[self.melanoma_idx]
            if mel_class in classes:
                mel_idx = classes.index(mel_class)
                for ax in axes.flat:
                    bars = ax.patches
                    if mel_idx < len(bars):
                        bars[mel_idx].set_edgecolor('red')
                        bars[mel_idx].set_linewidth(3)
        
        plt.suptitle('Performance Metrics by Class - Skin Cancer Classification', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save_figs:
            plt.savefig(f'{self.figures_dir}/class_metrics.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{self.figures_dir}/class_metrics.pdf', bbox_inches='tight')
        plt.show()
    
    def _plot_melanoma_analysis(self, metrics, save_figs):
        """Análisis específico de melanoma"""
        if not metrics['melanoma']:
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Métricas de melanoma
        mel_metrics = ['melanoma_recall', 'melanoma_precision', 'melanoma_f1', 'melanoma_specificity']
        mel_values = [metrics['melanoma'][m] for m in mel_metrics]
        mel_labels = ['Sensitivity', 'Precision', 'F1-Score', 'Specificity']
        
        bars = axes[0].bar(mel_labels, mel_values, color=['red', 'orange', 'green', 'blue'], alpha=0.8)
        axes[0].set_title('Melanoma Detection Performance', fontweight='bold', fontsize=14)
        axes[0].set_ylabel('Performance (%)')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 100)
        
        # Añadir valores en las barras
        for bar, value in zip(bars, mel_values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Comparación maligno vs benigno
        mb_metrics = metrics['malignant_benign']
        categories = ['Malignant\nSensitivity', 'Malignant\nSpecificity', 'Malignant\nPrecision', 'Malignant\nF1']
        mb_values = [mb_metrics['malignant_sensitivity'], mb_metrics['malignant_specificity'], 
                    mb_metrics['malignant_precision'], mb_metrics['malignant_f1']]
        
        bars2 = axes[1].bar(categories, mb_values, color=['darkred', 'darkblue', 'darkgreen', 'purple'], alpha=0.8)
        axes[1].set_title('Malignant vs Benign Classification', fontweight='bold', fontsize=14)
        axes[1].set_ylabel('Performance (%)')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 100)
        
        # Añadir valores en las barras
        for bar, value in zip(bars2, mb_values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        if save_figs:
            plt.savefig(f'{self.figures_dir}/melanoma_analysis.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{self.figures_dir}/melanoma_analysis.pdf', bbox_inches='tight')
        plt.show()
    
    def _plot_class_distribution(self, metrics, save_figs):
        """Distribución de clases en el dataset"""
        plt.figure(figsize=(12, 8))
        
        classes = list(metrics['by_class'].keys())
        supports = [metrics['by_class'][c]['support'] for c in classes]
        percentages = [s/sum(supports)*100 for s in supports]
        
        # Crear pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
        wedges, texts, autotexts = plt.pie(supports, 
                                          labels=classes, 
                                          colors=colors,
                                          autopct='%1.1f%%',
                                          startangle=90,
                                          explode=[0.1 if 'mel' in c.lower() else 0 for c in classes])
        
        # Destacar melanoma
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        plt.title('Class Distribution in Validation Dataset\nSkin Cancer Classification', 
                 fontsize=16, fontweight='bold')
        
        # Añadir leyenda con números absolutos
        legend_labels = [f'{c}: {s} samples ({p:.1f}%)' for c, s, p in zip(classes, supports, percentages)]
        plt.legend(wedges, legend_labels, title="Classes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        if save_figs:
            plt.savefig(f'{self.figures_dir}/class_distribution.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{self.figures_dir}/class_distribution.pdf', bbox_inches='tight')
        plt.show()
    
    def _plot_executive_summary(self, metrics, save_figs):
        """Resumen ejecutivo para artículo"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Métricas principales (grande)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        # Título principal
        ax1.text(0.5, 0.9, 'SKIN CANCER CLASSIFICATION MODEL - PERFORMANCE SUMMARY', 
                ha='center', va='center', fontsize=20, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        # Métricas clave
        basic = metrics['basic']
        mel = metrics['melanoma'] if metrics['melanoma'] else {}
        mb = metrics['malignant_benign']
        
        summary_text = f"""
        OVERALL ACCURACY: {basic['accuracy']:.1f}%        BALANCED ACCURACY: {basic['balanced_accuracy']:.1f}%        TOTAL SAMPLES: {basic['total_samples']:,}
        
        MELANOMA DETECTION: Sensitivity {mel.get('melanoma_recall', 0):.1f}% | Precision {mel.get('melanoma_precision', 0):.1f}% | F1-Score {mel.get('melanoma_f1', 0):.1f}%
        
        MALIGNANT vs BENIGN: Sensitivity {mb['malignant_sensitivity']:.1f}% | Specificity {mb['malignant_specificity']:.1f}%
        """
        
        ax1.text(0.5, 0.3, summary_text, ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        # Mini gráficos
        # ROC curves (mini)
        ax2 = fig.add_subplot(gs[1, 0])
        if metrics['roc_data']:
            for class_name, data in metrics['roc_data'].items():
                ax2.plot(data['fpr'], data['tpr'], lw=2, label=f'{class_name} ({data["auc"]:.2f})')
            ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax2.set_title('ROC Curves', fontweight='bold')
            ax2.set_xlabel('FPR')
            ax2.set_ylabel('TPR')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=8)
        
        # Métricas por clase (mini)
        ax3 = fig.add_subplot(gs[1, 1])
        classes = list(metrics['by_class'].keys())
        f1_scores = [metrics['by_class'][c]['f1_score'] for c in classes]
        bars = ax3.bar(range(len(classes)), f1_scores, color=self.colors[:len(classes)])
        ax3.set_title('F1-Scores by Class', fontweight='bold')
        ax3.set_ylabel('F1-Score (%)')
        ax3.set_xticks(range(len(classes)))
        ax3.set_xticklabels(classes, rotation=45, fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Matriz de confusión (mini)
        ax4 = fig.add_subplot(gs[1, 2])
        cm_norm = metrics['confusion_matrix'].astype('float') / metrics['confusion_matrix'].sum(axis=1)[:, np.newaxis]
        im = ax4.imshow(cm_norm, cmap='Blues')
        ax4.set_title('Confusion Matrix', fontweight='bold')
        ax4.set_xticks(range(len(classes)))
        ax4.set_yticks(range(len(classes)))
        ax4.set_xticklabels(classes, rotation=45, fontsize=8)
        ax4.set_yticklabels(classes, fontsize=8)
        
        # Tabla de resultados
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # Crear tabla
        table_data = []
        for class_name in classes:
            cm = metrics['by_class'][class_name]
            table_data.append([
                class_name.upper(),
                f"{cm['recall']:.1f}%",
                f"{cm['precision']:.1f}%",
                f"{cm['f1_score']:.1f}%",
                f"{cm['specificity']:.1f}%",
                f"{cm['support']}"
            ])
        
        table = ax5.table(cellText=table_data,
                         colLabels=['Class', 'Sensitivity', 'Precision', 'F1-Score', 'Specificity', 'Support'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0.1, 0.1, 0.8, 0.8])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Colorear header
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Timestamp
        ax5.text(0.95, 0.02, f"Generated: {metrics['timestamp']}", 
                ha='right', va='bottom', fontsize=10, style='italic',
                transform=ax5.transAxes)
        
        if save_figs:
            plt.savefig(f'{self.figures_dir}/executive_summary.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{self.figures_dir}/executive_summary.pdf', bbox_inches='tight')
        plt.show()
    
    def print_medical_report(self, metrics):
        """Reporte médico completo para artículo"""
        print(f"\n{'='*100}")
        print("MEDICAL PERFORMANCE REPORT - SKIN CANCER CLASSIFICATION MODEL")
        print(f"{'='*100}")
        print(f"Generated: {metrics['timestamp']}")
        print(f"Total Validation Samples: {metrics['basic']['total_samples']:,}")
        
        # MÉTRICAS GENERALES
        basic = metrics['basic']
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   • Overall Accuracy: {basic['accuracy']:.2f}%")
        print(f"   • Balanced Accuracy: {basic['balanced_accuracy']:.2f}%")
        
        # MELANOMA (CRÍTICO PARA ARTÍCULO)
        if metrics['melanoma']:
            mel = metrics['melanoma']
            print(f"\n🔴 MELANOMA DETECTION (CRITICAL):")
            print(f"   • Sensitivity (Recall): {mel['melanoma_recall']:.2f}%")
            print(f"   • Specificity: {mel['melanoma_specificity']:.2f}%")
            print(f"   • Precision (PPV): {mel['melanoma_precision']:.2f}%")
            print(f"   • F1-Score: {mel['melanoma_f1']:.2f}%")
            print(f"   • Support: {mel['melanoma_support']} samples")
            
            # Interpretación clínica
            if mel['melanoma_recall'] >= 90:
                print(f"   ✅ EXCELLENT: High sensitivity reduces false negatives")
            elif mel['melanoma_recall'] >= 80:
                print(f"   ✅ GOOD: Acceptable sensitivity for screening")
            else:
                print(f"   ⚠️ CONCERN: Low sensitivity may miss melanomas")
        
        # MALIGNO vs BENIGNO
        mb = metrics['malignant_benign']
        print(f"\n⚡ MALIGNANT vs BENIGN CLASSIFICATION:")
        print(f"   • Malignant Sensitivity: {mb['malignant_sensitivity']:.2f}%")
        print(f"   • Malignant Specificity: {mb['malignant_specificity']:.2f}%")
        print(f"   • Malignant Precision: {mb['malignant_precision']:.2f}%")
        print(f"   • Malignant F1-Score: {mb['malignant_f1']:.2f}%")
        print(f"   • Malignant Support: {mb['malignant_support']} samples")
        print(f"   • Benign Support: {mb['benign_support']} samples")
        
        # AUC SCORES
        if metrics['auc']:
            auc = metrics['auc']
            print(f"\n📈 AUC SCORES (Area Under ROC Curve):")
            print(f"   • AUC Macro Average: {auc.get('auc_macro', 0):.3f}")
            print(f"   • AUC Weighted Average: {auc.get('auc_weighted', 0):.3f}")
            
            print(f"\n   📋 AUC by Class:")
            for class_name in self.class_names:
                auc_key = f'auc_{class_name}'
                if auc_key in auc:
                    print(f"      • {class_name.upper()}: {auc[auc_key]:.3f}")
        
        # DETALLE POR CLASE
        print(f"\n📋 DETAILED PERFORMANCE BY CLASS:")
        print(f"{'Class':<8} {'Sensitivity':<12} {'Precision':<12} {'F1-Score':<10} {'Specificity':<12} {'Support':<8}")
        print("-" * 70)
        
        for class_name, class_metrics in metrics['by_class'].items():
            print(f"{class_name.upper():<8} "
                  f"{class_metrics['recall']:<12.1f} "
                  f"{class_metrics['precision']:<12.1f} "
                  f"{class_metrics['f1_score']:<10.1f} "
                  f"{class_metrics['specificity']:<12.1f} "
                  f"{class_metrics['support']:<8}")
        
        # RECOMENDACIONES CLÍNICAS
        print(f"\n💡 CLINICAL RECOMMENDATIONS:")
        
        if metrics['melanoma']:
            mel_recall = metrics['melanoma']['melanoma_recall']
            if mel_recall < 85:
                print(f"   ⚠️ Consider increasing melanoma sensitivity threshold")
            if mel_recall >= 90:
                print(f"   ✅ Melanoma detection suitable for screening applications")
        
        if basic['accuracy'] >= 85:
            print(f"   ✅ Overall accuracy suitable for clinical decision support")
        else:
            print(f"   ⚠️ Consider model improvement before clinical deployment")
        
        if mb['malignant_sensitivity'] >= 85:
            print(f"   ✅ Good malignant lesion detection capability")
        else:
            print(f"   ⚠️ May miss some malignant lesions - use with caution")
        
        print(f"\n{'='*100}")
        print("END OF MEDICAL REPORT")
        print(f"{'='*100}")
    
    def export_results_to_csv(self, metrics, filename="model_performance_results.csv"):
        """Exportar resultados a CSV para artículo"""
        
        # Preparar datos para CSV
        results_data = []
        
        # Métricas generales
        results_data.append({
            'Metric_Type': 'Overall',
            'Class': 'All',
            'Sensitivity_Recall': '',
            'Precision': '',
            'F1_Score': '',
            'Specificity': '',
            'Support': metrics['basic']['total_samples'],
            'Accuracy': metrics['basic']['accuracy'],
            'AUC': metrics['auc'].get('auc_macro', '') if metrics['auc'] else ''
        })
        
        # Métricas por clase
        for class_name, class_metrics in metrics['by_class'].items():
            auc_value = metrics['auc'].get(f'auc_{class_name}', '') if metrics['auc'] else ''
            
            results_data.append({
                'Metric_Type': 'By_Class',
                'Class': class_name.upper(),
                'Sensitivity_Recall': round(class_metrics['recall'], 2),
                'Precision': round(class_metrics['precision'], 2),
                'F1_Score': round(class_metrics['f1_score'], 2),
                'Specificity': round(class_metrics['specificity'], 2),
                'Support': class_metrics['support'],
                'Accuracy': '',
                'AUC': round(auc_value, 3) if auc_value else ''
            })
        
        # Maligno vs Benigno
        mb = metrics['malignant_benign']
        results_data.append({
            'Metric_Type': 'Malignant_vs_Benign',
            'Class': 'Malignant',
            'Sensitivity_Recall': round(mb['malignant_sensitivity'], 2),
            'Precision': round(mb['malignant_precision'], 2),
            'F1_Score': round(mb['malignant_f1'], 2),
            'Specificity': round(mb['malignant_specificity'], 2),
            'Support': mb['malignant_support'],
            'Accuracy': '',
            'AUC': ''
        })
        
        # Crear DataFrame y guardar
        df = pd.DataFrame(results_data)
        csv_path = f"{self.figures_dir}/{filename}"
        df.to_csv(csv_path, index=False)
        
        print(f"✅ Resultados exportados a: {csv_path}")
        return csv_path