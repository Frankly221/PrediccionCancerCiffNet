import subprocess
import sys
import time
import os
import torch

def main():
    """Ejecutar entrenamiento mejorado"""
    start_time = time.time()
    
    print("🚀 ENTRENAMIENTO CIFF-NET MEJORADO")
    print("=" * 70)
    print("🎯 Mejoras implementadas:")
    print("   ✅ EfficientNet-B1 (más potente)")
    print("   ✅ Focal Loss para melanoma")
    print("   ✅ Oversample clases minoritarias")
    print("   ✅ Augmentación médica específica")
    print("   ✅ Multi-head attention mejorado")
    print("   ✅ SE blocks en todas las capas")
    print("   ✅ Scheduler adaptativo")
    print("   ✅ Tracking específico melanoma")
    print("   ✅ Múltiples criterios de guardado")
    print("=" * 70)
    
    # Limpiar GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Configurar entorno
    env = os.environ.copy()
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    env['PYTHONIOENCODING'] = 'utf-8'
    
    try:
        # Ejecutar entrenamiento mejorado
        result = subprocess.run(
            [sys.executable, 'train_improved.py'], 
            env=env,
            encoding='utf-8',
            errors='ignore'
        )
        
        if result.returncode == 0:
            print("🎉 Entrenamiento mejorado completado exitosamente!")
            
            # Verificar archivos generados
            expected_files = [
                'best_overall_improved.pth',
                'best_melanoma_improved.pth', 
                'best_balanced_improved.pth',
                'training_history_improved.png',
                'confusion_matrix_improved.png',
                'classification_report_improved.csv'
            ]
            
            print("\n📁 Archivos generados:")
            for file in expected_files:
                if os.path.exists(file):
                    size_mb = os.path.getsize(file) / (1024*1024)
                    print(f"   ✅ {file} ({size_mb:.1f} MB)")
                else:
                    print(f"   ❌ {file}")
            
            total_time = time.time() - start_time
            print(f"\n⏱️  Tiempo total: {total_time/3600:.2f} horas")
            print("\n🎯 Próximos pasos:")
            print("   1. Revisar training_history_improved.png")
            print("   2. Comparar confusion_matrix_improved.png con anterior")
            print("   3. Analizar mejora en detección de melanoma")
            print("   4. Usar best_melanoma_improved.pth para aplicaciones críticas")
            
        else:
            print("❌ Error en entrenamiento mejorado")
            
    except Exception as e:
        print(f"❌ Error ejecutando entrenamiento: {e}")

if __name__ == "__main__":
    main()