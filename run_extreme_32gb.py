import subprocess
import sys
import time
import os
import torch
import psutil

def check_system_requirements():
    """Verificar que el sistema puede manejar configuración extrema"""
    
    print("🔍 VERIFICANDO SISTEMA PARA CONFIGURACIÓN EXTREMA...")
    
    # Check RAM
    ram_info = psutil.virtual_memory()
    ram_gb = ram_info.total / 1e9
    
    if ram_gb < 30:
        print(f"⚠️  RAM: {ram_gb:.1f}GB - Recomendado: 32GB+")
        return False
    else:
        print(f"✅ RAM: {ram_gb:.1f}GB - PERFECTO")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA no disponible")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    if gpu_memory < 7:
        print(f"⚠️  GPU: {gpu_name} ({gpu_memory:.1f}GB) - Recomendado: 8GB+")
        return False
    else:
        print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB) - PERFECTO")
    
    # Check CPU
    cpu_count = psutil.cpu_count()
    if cpu_count < 12:
        print(f"⚠️  CPU: {cpu_count} cores - Recomendado: 12+")
        return False
    else:
        print(f"✅ CPU: {cpu_count} cores - PERFECTO")
    
    print("🚀 SISTEMA APTO PARA CONFIGURACIÓN EXTREMA!")
    return True

def main():
    """Ejecutar entrenamiento EXTREMO 32GB RAM"""
    start_time = time.time()
    
    print("🚀 CIFF-NET EXTREMO - 32GB RAM + RTX 3070")
    print("=" * 70)
    
    # Verificar sistema
    if not check_system_requirements():
        print("❌ Sistema no apto para configuración extrema")
        return
    
    print("\n🎯 OPTIMIZACIONES EXTREMAS APLICADAS:")
    print("   ✅ Batch size: 28 (MÁXIMO RTX 3070)")
    print("   ✅ VRAM usage: 99%")
    print("   ✅ RAM workers: 20")
    print("   ✅ Prefetch factor: 6")
    print("   ✅ CPU threads: 16")
    print("   ✅ TF32 habilitado")
    print("   ✅ Scheduler agresivo (patience=6)")
    print("   ✅ Memory cleanup optimizado")
    print("   ✅ Monitoring avanzado")
    print("=" * 70)
    
    # Configurar entorno EXTREMO
    env = os.environ.copy()
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,roundup_power2_divisions:16'
    env['PYTHONIOENCODING'] = 'utf-8'
    env['OMP_NUM_THREADS'] = '16'
    env['MKL_NUM_THREADS'] = '16'
    env['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Instalar psutil si no está instalado
    try:
        import psutil
    except ImportError:
        print("📦 Instalando psutil para monitoring...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'psutil'])
    
    try:
        print("\n🚀 Iniciando entrenamiento EXTREMO 32GB+RTX3070...")
        
        # Ejecutar entrenamiento EXTREMO
        result = subprocess.run(
            [sys.executable, 'train_max_gpu.py'], 
            env=env,
            encoding='utf-8',
            errors='ignore'
        )
        
        if result.returncode == 0:
            print("🎉 Entrenamiento EXTREMO completado exitosamente!")
            
            # Verificar archivos generados
            expected_files = [
                'best_32gb_rtx3070_overall.pth',
                'best_32gb_rtx3070_melanoma.pth', 
                'best_32gb_rtx3070_balanced.pth',
                'training_history_32gb_rtx3070.png',
                'training_metrics_32gb_rtx3070.csv'
            ]
            
            print("\n📁 Archivos EXTREMOS generados:")
            total_size = 0
            for file in expected_files:
                if os.path.exists(file):
                    size_mb = os.path.getsize(file) / (1024*1024)
                    total_size += size_mb
                    print(f"   ✅ {file} ({size_mb:.1f} MB)")
                else:
                    print(f"   ❌ {file}")
            
            print(f"\n💾 Tamaño total archivos: {total_size:.1f} MB")
            
            total_time = time.time() - start_time
            print(f"⏱️  Tiempo total: {total_time/3600:.2f} horas")
            print(f"🎯 GPU Usage esperado: 95-99%")
            print(f"💾 RAM Usage esperado: 60-80%")
            
            # Mostrar estadísticas finales
            try:
                import pandas as pd
                if os.path.exists('training_metrics_32gb_rtx3070.csv'):
                    df = pd.read_csv('training_metrics_32gb_rtx3070.csv')
                    print(f"\n📊 ESTADÍSTICAS FINALES:")
                    print(f"   Mejor Accuracy: {df['val_acc'].max():.2f}%")
                    print(f"   Mejor Melanoma Recall: {df['melanoma_recall'].max():.2f}%")
                    print(f"   VRAM Promedio: {df['gpu_memory_gb'].mean():.1f}GB")
                    print(f"   RAM Promedio: {df['ram_usage_percent'].mean():.1f}%")
                    print(f"   Tiempo Batch Promedio: {df['batch_time_sec'].mean():.3f}s")
            except:
                pass
            
        else:
            print("❌ Error en entrenamiento EXTREMO")
            
    except Exception as e:
        print(f"❌ Error ejecutando entrenamiento: {e}")

if __name__ == "__main__":
    main()