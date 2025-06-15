import subprocess
import sys
import time
import os
import torch

def check_gpu():
    """Verificar GPU disponible"""
    if torch.cuda.is_available():
        print(f"🔥 GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("❌ No se detectó GPU CUDA")
        return False

def check_dataset():
    """Verificar que el dataset existe"""
    csv_file = "datasetHam10000/HAM10000_metadata.csv"
    image_folders = [
        "datasetHam10000/HAM10000_images_part_1", 
        "datasetHam10000/HAM10000_images_part_2"
    ]
    
    if not os.path.exists(csv_file):
        print(f"❌ No se encontró: {csv_file}")
        return False
    
    for folder in image_folders:
        if not os.path.exists(folder):
            print(f"❌ No se encontró: {folder}")
            return False
    
    print("✅ Dataset HAM10000 encontrado")
    return True

def run_phase(phase_name, script_name):
    """Ejecutar una fase del entrenamiento con manejo de encoding"""
    print(f"\n{'='*70}")
    print(f"FASE: {phase_name}")
    print(f"{'='*70}")
    
    try:
        # Configurar entorno
        env = os.environ.copy()
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        env['PYTHONIOENCODING'] = 'utf-8'  # Forzar UTF-8
        
        # Ejecutar script con encoding UTF-8
        result = subprocess.run(
            [sys.executable, script_name], 
            capture_output=True, 
            text=True,
            env=env,
            encoding='utf-8',  # Especificar encoding
            errors='ignore'    # Ignorar errores de encoding
        )
        
        if result.returncode == 0:
            print(f"✅ {phase_name} completada exitosamente")
            print("📤 Últimas líneas de salida:")
            # Mostrar últimas líneas sin problemas de encoding
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[-10:]:  # Últimas 10 líneas
                try:
                    print(f"   {line}")
                except:
                    print("   [línea con caracteres especiales]")
            return True
        else:
            print(f"❌ Error en {phase_name}")
            print("📤 Error encontrado:")
            # Mostrar error sin problemas de encoding
            error_lines = result.stderr.strip().split('\n')
            for line in error_lines[-15:]:  # Últimas 15 líneas de error
                try:
                    print(f"   {line}")
                except:
                    print("   [línea de error con caracteres especiales]")
            return False
            
    except Exception as e:
        print(f"❌ Error ejecutando {phase_name}: {e}")
        return False

def main():
    """Función principal de entrenamiento completo"""
    start_time = time.time()
    
    print("🚀 ENTRENAMIENTO COMPLETO CIFF-NET RTX 3070 Ti")
    print("=" * 70)
    
    # Verificaciones previas
    if not check_gpu():
        print("⚠️  Continuando sin GPU (usará CPU)")
    
    if not check_dataset():
        print("❌ Dataset no encontrado. Verifica las rutas.")
        return
    
    # Limpiar cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 Cache GPU limpiado")
    
    # Ejecutar Fase 1: CIFF-Net optimizado
    phases_completed = 0
    total_phases = 1
    
    print(f"\n🎯 Iniciando entrenamiento optimizado para RTX 3070 Ti...")
    print(f"🔧 Configuraciones aplicadas:")
    print(f"   - Batch size: 16")
    print(f"   - Mixed Precision: Habilitado")
    print(f"   - Gradient Checkpointing: Habilitado")
    print(f"   - Memory Management: Optimizado")
    print(f"   - MKSA: Solo última capa")
    print(f"   - Backbone: EfficientNet-B0")
    
    # Fase 1: Entrenamiento principal
    if run_phase("CIFF-Net RTX 8GB - Dimensiones Corregidas", "train.py"):
        phases_completed += 1
        print("🎉 Fase 1 completada!")
    else:
        print("❌ Fase 1 falló. Revisa el error arriba.")
        print("💡 Sugerencias:")
        print("   - Verifica que el dataset esté completo")
        print("   - Asegúrate de tener al menos 4GB VRAM libres")
        print("   - Prueba reducir batch_size a 8 si persiste el error")
        return
    
    # Resumen final
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print("📊 RESUMEN DEL ENTRENAMIENTO CIFF-NET RTX")
    print(f"{'='*70}")
    print(f"⏱️  Tiempo total: {total_time/3600:.2f} horas")
    print(f"✅ Fases completadas: {phases_completed}/{total_phases}")
    
    if phases_completed == total_phases:
        print("🎉 ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
        print("\n📁 Archivos generados:")
        
        files_to_check = [
            "best_ciff_net_rtx8gb.pth",
            "training_history_rtx8gb.png", 
            "confusion_matrix_rtx8gb.png",
            "classification_report_rtx8gb.csv"
        ]
        
        for file in files_to_check:
            if os.path.exists(file):
                size_mb = os.path.getsize(file) / (1024*1024)
                print(f"   ✅ {file} ({size_mb:.1f} MB)")
            else:
                print(f"   ❌ {file} (no encontrado)")
                
        print("\n🔥 Tu RTX 3070 Ti ha entrenado CIFF-Net exitosamente!")
        print("🎯 Próximos pasos:")
        print("   1. Revisar training_history_rtx8gb.png para ver curvas")
        print("   2. Analizar confusion_matrix_rtx8gb.png para ver rendimiento")
        print("   3. Usar best_ciff_net_rtx8gb.pth para predicciones")
        
    else:
        print("⚠️  Entrenamiento incompleto")
        print("💡 Revisa los errores arriba y reintenta")
    
    print("=" * 70)

if __name__ == "__main__":
    main()