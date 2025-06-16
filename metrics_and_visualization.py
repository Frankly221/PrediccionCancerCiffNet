import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score,
    roc_auc_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class CiffNetMetrics:
    """
    Clase completa para métricas y visualizaciones de CiffNet
    Optimizada para papers científicos de dermatología IA
    """
    
    def __init__(self, num_classes=7, class_names=None, save_dir="results"):
        self.num_classes = num_classes
        self.save_dir = save_dir
        
        # Nombres de clases HAM10000
        if class_names is None:
            self.class_names = [
                'MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC'
            ]
        else:
            self.class_names = class_names
        
        # Crear directorios
        self._create_directories()
        
        print(f"📊 CiffNet Metrics inicializado:")
        print(f"   Classes: {self.num_classes}")
        print(f"   Save dir: {self.save_dir}")
        print(f"   Class names: {self.class_names}")
    
    def _create_directories(self):
        """Crear directorios para guardar resultados"""
        dirs = ['models', 'metrics', 'visualizations', 'analysis']
        for dir_name in dirs:
            os.makedirs(f"{self.save_dir}/{dir_name}", exist_ok=True)
    
    def compute_basic_metrics(self, y_true, y_pred, y_probs=None):
        """
        Métricas básicas de clasificación
        """
        # Convertir a numpy si es tensor
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        if torch.is_tensor(y_probs):
            y_probs = y_probs.cpu().numpy()
        
        # Métricas básicas
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Métricas adicionales
        kappa = cohen_kappa_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'cohen_kappa': float(kappa),
            'matthews_corrcoef': float(mcc),
            'balanced_accuracy': float(balanced_acc)
        }
        
        # Métricas por clase
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            class_metrics[class_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1': float(f1_per_class[i])
            }
        
        metrics['per_class'] = class_metrics
        
        # Métricas probabilísticas si disponibles
        if y_probs is not None:
            try:
                # AUC multiclass
                auc_macro = roc_auc_score(y_true, y_probs, average='macro', multi_class='ovr')
                auc_weighted = roc_auc_score(y_true, y_probs, average='weighted', multi_class='ovr')
                
                # Brier score
                y_true_onehot = np.eye(self.num_classes)[y_true]
                brier = brier_score_loss(y_true_onehot.ravel(), y_probs.ravel())
                
                metrics.update({
                    'auc_macro': float(auc_macro),
                    'auc_weighted': float(auc_weighted),
                    'brier_score': float(brier)
                })
            except Exception as e:
                print(f"⚠️ Warning computing probabilistic metrics: {e}")
        
        return metrics
    
    def compute_cliff_metrics(self, cliff_scores, cliff_targets, y_true, y_pred, threshold=0.15):
        """
        Métricas específicas para cliff detection
        """
        if torch.is_tensor(cliff_scores):
            cliff_scores = cliff_scores.cpu().numpy()
        if torch.is_tensor(cliff_targets):
            cliff_targets = cliff_targets.cpu().numpy()
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        
        # Cliff detection como clasificación binaria
        cliff_pred = (cliff_scores > threshold).astype(int)
        
        # Si no hay cliff_targets, usar accuracy de predicción como proxy
        if cliff_targets is None:
            cliff_targets = (y_true != y_pred).astype(int)  # Muestras mal clasificadas como cliff
        
        # Métricas cliff detection
        cliff_accuracy = accuracy_score(cliff_targets, cliff_pred)
        cliff_precision = precision_score(cliff_targets, cliff_pred, zero_division=0)
        cliff_recall = recall_score(cliff_targets, cliff_pred, zero_division=0)
        cliff_f1 = f1_score(cliff_targets, cliff_pred, zero_division=0)
        
        # Análisis por grupos
        cliff_mask = cliff_pred.astype(bool)
        non_cliff_mask = ~cliff_mask
        
        # Performance en cliff vs non-cliff
        if cliff_mask.sum() > 0:
            cliff_accuracy_subset = accuracy_score(y_true[cliff_mask], y_pred[cliff_mask])
        else:
            cliff_accuracy_subset = 0.0
            
        if non_cliff_mask.sum() > 0:
            non_cliff_accuracy_subset = accuracy_score(y_true[non_cliff_mask], y_pred[non_cliff_mask])
        else:
            non_cliff_accuracy_subset = 0.0
        
        # Estadísticas cliff scores
        cliff_stats = {
            'mean': float(np.mean(cliff_scores)),
            'std': float(np.std(cliff_scores)),
            'min': float(np.min(cliff_scores)),
            'max': float(np.max(cliff_scores)),
            'median': float(np.median(cliff_scores)),
            'q25': float(np.percentile(cliff_scores, 25)),
            'q75': float(np.percentile(cliff_scores, 75))
        }
        
        cliff_metrics = {
            'cliff_detection_accuracy': float(cliff_accuracy),
            'cliff_detection_precision': float(cliff_precision),
            'cliff_detection_recall': float(cliff_recall),
            'cliff_detection_f1': float(cliff_f1),
            'cliff_ratio': float(cliff_mask.mean()),
            'performance_on_cliff': float(cliff_accuracy_subset),
            'performance_on_non_cliff': float(non_cliff_accuracy_subset),
            'cliff_score_stats': cliff_stats,
            'samples_identified_as_cliff': int(cliff_mask.sum()),
            'total_samples': len(cliff_scores)
        }
        
        return cliff_metrics
    
    def compute_confidence_metrics(self, confidences, y_true, y_pred, n_bins=10):
        """
        Métricas de calibración y confianza
        """
        if torch.is_tensor(confidences):
            confidences = confidences.cpu().numpy()
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        
        # Accuracy per prediction
        correct = (y_true == y_pred).astype(int)
        
        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Estadísticas confianza
        conf_stats = {
            'mean': float(np.mean(confidences)),
            'std': float(np.std(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences)),
            'median': float(np.median(confidences))
        }
        
        # Correlación confianza-accuracy
        try:
            conf_accuracy_corr = np.corrcoef(confidences, correct)[0, 1]
        except:
            conf_accuracy_corr = 0.0
        
        confidence_metrics = {
            'expected_calibration_error': float(ece),
            'confidence_stats': conf_stats,
            'confidence_accuracy_correlation': float(conf_accuracy_corr),
            'n_bins': n_bins
        }
        
        return confidence_metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, normalize=True, save_name="confusion_matrix"):
        """
        Plot confusion matrix de alta calidad
        """
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        
        # Calcular confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_display = cm_norm
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            cm_display = cm
            fmt = 'd'
            title = 'Confusion Matrix'
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_display, 
                   annot=True, 
                   fmt=fmt, 
                   cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Proportion' if normalize else 'Count'})
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Guardar
        save_path = f"{self.save_dir}/visualizations/{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Confusion matrix guardada: {save_path}")
        return cm
    
    def plot_roc_curves(self, y_true, y_probs):
        """
        Plot ROC curves para cada clase - CORREGIDO para manejar NaN
        """
        try:
            # ✅ VALIDACIÓN Y LIMPIEZA DE DATOS
            print("🔍 Validando datos para ROC curves...")
            
            # Verificar y limpiar NaN en probabilidades
            nan_mask = np.isnan(y_probs).any(axis=1)
            if nan_mask.any():
                print(f"⚠️ Encontrados {nan_mask.sum()} samples con NaN, eliminando...")
                y_probs = y_probs[~nan_mask]
                y_true = y_true[~nan_mask]
            
            # Verificar que no hay infinitos
            inf_mask = np.isinf(y_probs).any(axis=1)
            if inf_mask.any():
                print(f"⚠️ Encontrados {inf_mask.sum()} samples con infinitos, eliminando...")
                y_probs = y_probs[~inf_mask]
                y_true = y_true[~inf_mask]
            
            # Verificar que las probabilidades están en rango válido [0,1]
            if y_probs.min() < 0 or y_probs.max() > 1:
                print(f"⚠️ Probabilidades fuera de rango [0,1], normalizando...")
                # Aplicar softmax para normalizar
                y_probs = np.exp(y_probs) / np.sum(np.exp(y_probs), axis=1, keepdims=True)
            
            # Verificar que tenemos datos suficientes
            if len(y_true) < 10:
                print(f"❌ Datos insuficientes después de limpieza ({len(y_true)} samples)")
                return
            
            print(f"✅ Datos validados: {len(y_true)} samples, {y_probs.shape[1]} clases")
            
            # Binarizar labels para ROC multiclase
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
            
            # Si solo hay una clase, label_binarize devuelve array 1D
            if y_true_bin.ndim == 1:
                y_true_bin = y_true_bin.reshape(-1, 1)
            
            # Crear figura
            plt.figure(figsize=(12, 10))
            colors = plt.cm.Set3(np.linspace(0, 1, self.num_classes))
            
            # ROC para cada clase
            roc_auc = {}
            all_fpr = []
            all_tpr = []
            
            for i, (color, class_name) in enumerate(zip(colors, self.class_names)):
                # ✅ VALIDACIÓN ADICIONAL POR CLASE
                if i >= y_probs.shape[1]:
                    print(f"⚠️ Saltando clase {i}: índice fuera de rango")
                    continue
                    
                # Verificar que la clase tiene samples
                if i >= y_true_bin.shape[1]:
                    print(f"⚠️ Saltando clase {i}: no hay datos binarizados")
                    continue
                
                class_y_true = y_true_bin[:, i]
                class_y_probs = y_probs[:, i]
                
                # Verificar NaN específicos de esta clase
                class_nan_mask = np.isnan(class_y_probs) | np.isnan(class_y_true)
                if class_nan_mask.any():
                    print(f"⚠️ Clase {class_name}: {class_nan_mask.sum()} valores NaN, limpiando...")
                    class_y_true = class_y_true[~class_nan_mask]
                    class_y_probs = class_y_probs[~class_nan_mask]
                
                # Verificar que hay al menos dos clases (0 y 1) en y_true
                if len(np.unique(class_y_true)) < 2:
                    print(f"⚠️ Clase {class_name}: solo una clase presente, saltando ROC")
                    continue
                
                try:
                    # ✅ CÁLCULO SEGURO DE ROC
                    from sklearn.metrics import roc_curve, auc
                    fpr, tpr, _ = roc_curve(class_y_true, class_y_probs)
                    roc_auc[i] = auc(fpr, tpr)
                    
                    # Plot individual
                    plt.plot(fpr, tpr, color=color, lw=2,
                            label=f'{class_name} (AUC = {roc_auc[i]:.3f})')
                    
                    all_fpr.append(fpr)
                    all_tpr.append(tpr)
                    
                except Exception as e:
                    print(f"⚠️ Error en ROC para clase {class_name}: {e}")
                    continue
            
            # ROC curve promedio (micro-average)
            try:
                from sklearn.metrics import roc_curve, auc
                fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
                roc_auc_micro = auc(fpr_micro, tpr_micro)
                
                plt.plot(fpr_micro, tpr_micro,
                        color='deeppink', linestyle=':', linewidth=4,
                        label=f'Micro-average (AUC = {roc_auc_micro:.3f})')
            except Exception as e:
                print(f"⚠️ Error en micro-average ROC: {e}")
            
            # Línea diagonal (random classifier)
            plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
            
            # Configuración del plot
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, fontweight='bold')
            plt.legend(loc="lower right", fontsize=10)
            plt.grid(True, alpha=0.3)
            
            # Guardar
            save_path = f"{self.save_dir}/visualizations/roc_curves.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ ROC curves guardadas: {save_path}")
            
            return roc_auc
            
        except Exception as e:
            print(f"❌ Error crítico en plot_roc_curves: {e}")
            print("🔧 Creando ROC curve simplificado...")
            
            # ✅ FALLBACK: ROC SIMPLIFICADO
            try:
                plt.figure(figsize=(8, 6))
                plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curves (Error en datos - Fallback)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                save_path = f"{self.save_dir}/visualizations/roc_curves_fallback.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"✅ ROC fallback guardado: {save_path}")
                
            except Exception as e2:
                print(f"❌ Error en fallback ROC: {e2}")
            
            return {}

    def plot_precision_recall_curves(self, y_true, y_probs):
        """
        Plot Precision-Recall curves - CORREGIDO para manejar NaN
        """
        try:
            # ✅ VALIDACIÓN Y LIMPIEZA DE DATOS (mismo proceso que ROC)
            print("🔍 Validando datos para Precision-Recall curves...")
            
            # Limpiar NaN
            nan_mask = np.isnan(y_probs).any(axis=1)
            if nan_mask.any():
                print(f"⚠️ Eliminando {nan_mask.sum()} samples con NaN...")
                y_probs = y_probs[~nan_mask]
                y_true = y_true[~nan_mask]
            
            # Limpiar infinitos
            inf_mask = np.isinf(y_probs).any(axis=1)
            if inf_mask.any():
                print(f"⚠️ Eliminando {inf_mask.sum()} samples con infinitos...")
                y_probs = y_probs[~inf_mask]
                y_true = y_true[~inf_mask]
            
            # Normalizar probabilidades
            if y_probs.min() < 0 or y_probs.max() > 1:
                print(f"⚠️ Normalizando probabilidades...")
                y_probs = np.exp(y_probs) / np.sum(np.exp(y_probs), axis=1, keepdims=True)
            
            if len(y_true) < 10:
                print(f"❌ Datos insuficientes para PR curves ({len(y_true)} samples)")
                return
            
            # Binarizar labels
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
            if y_true_bin.ndim == 1:
                y_true_bin = y_true_bin.reshape(-1, 1)
            
            # Crear figura
            plt.figure(figsize=(12, 10))
            colors = plt.cm.Set3(np.linspace(0, 1, self.num_classes))
            
            # PR curve para cada clase
            pr_auc = {}
            
            for i, (color, class_name) in enumerate(zip(colors, self.class_names)):
                if i >= y_probs.shape[1] or i >= y_true_bin.shape[1]:
                    continue
                    
                class_y_true = y_true_bin[:, i]
                class_y_probs = y_probs[:, i]
                
                # Limpiar NaN específicos
                class_nan_mask = np.isnan(class_y_probs) | np.isnan(class_y_true)
                if class_nan_mask.any():
                    class_y_true = class_y_true[~class_nan_mask]
                    class_y_probs = class_y_probs[~class_nan_mask]
                
                # Verificar variedad de clases
                if len(np.unique(class_y_true)) < 2:
                    print(f"⚠️ Clase {class_name}: solo una clase presente, saltando PR")
                    continue
                
                try:
                    from sklearn.metrics import precision_recall_curve, auc
                    precision, recall, _ = precision_recall_curve(class_y_true, class_y_probs)
                    pr_auc[i] = auc(recall, precision)
                    
                    plt.plot(recall, precision, color=color, lw=2,
                            label=f'{class_name} (AUC = {pr_auc[i]:.3f})')
                            
                except Exception as e:
                    print(f"⚠️ Error en PR para clase {class_name}: {e}")
                    continue
            
            # Configuración del plot
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall', fontsize=12)
            plt.ylabel('Precision', fontsize=12)
            plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
            plt.legend(loc="lower left", fontsize=10)
            plt.grid(True, alpha=0.3)
            
            # Guardar
            save_path = f"{self.save_dir}/visualizations/precision_recall_curves.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ PR curves guardadas: {save_path}")
            return pr_auc
            
        except Exception as e:
            print(f"❌ Error en Precision-Recall curves: {e}")
            return {}

    def save_metrics_json(self, metrics_dict, filename="detailed_metrics.json"):
        """
        Guardar métricas en JSON
        """
        # Agregar timestamp
        metrics_dict['timestamp'] = datetime.now().isoformat()
        metrics_dict['num_classes'] = self.num_classes
        metrics_dict['class_names'] = self.class_names
        
        save_path = f"{self.save_dir}/metrics/{filename}"
        with open(save_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        
        print(f"✅ Métricas guardadas: {save_path}")
    
    def generate_classification_report(self, y_true, y_pred, save_name="classification_report.txt"):
        """
        Generar reporte detallado de clasificación
        """
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            digits=4
        )
        
        save_path = f"{self.save_dir}/metrics/{save_name}"
        with open(save_path, 'w') as f:
            f.write("CIFFNET CLASSIFICATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)
            f.write(f"\n\nGenerated: {datetime.now().isoformat()}\n")
        
        print(f"✅ Classification report guardado: {save_path}")
        return report

def create_all_visualizations(metrics_calculator, y_true, y_pred, y_probs, 
                            cliff_scores=None, confidences=None, history=None):
    """
    Función helper para crear todas las visualizaciones
    """
    print("🎨 Generando todas las visualizaciones...")
    
    # Confusion Matrix
    metrics_calculator.plot_confusion_matrix(y_true, y_pred, normalize=True)
    metrics_calculator.plot_confusion_matrix(y_true, y_pred, normalize=False, save_name="confusion_matrix_counts")
    
    # ROC Curves
    if y_probs is not None:
        metrics_calculator.plot_roc_curves(y_true, y_probs)
        metrics_calculator.plot_precision_recall_curves(y_true, y_probs)
    
    # Training curves
    if history is not None:
        metrics_calculator.plot_training_curves(history)
    
    print("✅ Todas las visualizaciones generadas")

if __name__ == "__main__":
    # Test básico
    print("🧪 Testing CiffNet Metrics")
    
    # Datos de prueba
    np.random.seed(42)
    n_samples = 1000
    n_classes = 7
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_probs = np.random.rand(n_samples, n_classes)
    y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)  # Normalize
    
    # Crear metrics calculator
    metrics_calc = CiffNetMetrics(num_classes=n_classes)
    
    # Compute metrics
    basic_metrics = metrics_calc.compute_basic_metrics(y_true, y_pred, y_probs)
    print("✅ Basic metrics computed")
    
    # Generate visualizations
    create_all_visualizations(metrics_calc, y_true, y_pred, y_probs)
    
    # Save metrics
    metrics_calc.save_metrics_json(basic_metrics)
    metrics_calc.generate_classification_report(y_true, y_pred)
    
    print("🎯 CiffNet Metrics test completado")