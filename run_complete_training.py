import os
import time
import subprocess
import sys

def run_phase(phase_name, script_path, description):
    """Ejecutar una fase del entrenamiento"""
    print(f"\n{'='*80}")
    print(f"🚀 INICIANDO {phase_name.upper()}")
    print(f"📝 {description}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=False, text=True, check=True)
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ {phase_name} completada exitosamente en {elapsed_time/3600:.2f} horas")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error en {phase_name}: {e}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Archivo no encontrado: {script_path}")
        return False

def check_prerequisites():
    """Verificar prerequisitos antes del entrenamiento"""
    print("🔍 Verificando prerequisitos...")
    
    # Verificar archivos necesarios
    required_files = [
        'train.py',
        'train_phase2.py', 
        'train_phase3.py',
        'model.py',
        'model_phase2.py',
        'model_phase3.py',
        'dataset.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Archivos faltantes: {missing_files}")
        return False
    
    # Verificar dataset con rutas corregidas
    dataset_paths = [
        "datasetHam10000/HAM10000_metadata.csv",
        "datasetHam10000/HAM10000_images_part_1",  # Cambio aquí
        "datasetHam10000/HAM10000_images_part_2"   # Cambio aquí
    ]
    
    for path in dataset_paths:
        if not os.path.exists(path):
            print(f"❌ Dataset no encontrado: {path}")
            return False
    
    print("✅ Todos los prerequisitos cumplidos")
    return True

def main():
    """Ejecutar entrenamiento completo de CIFF-Net (3 fases)"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                         CIFF-NET TRAINING                        ║
    ║                    Cross-Image Feature Fusion Network            ║
    ║                                                                  ║
    ║  Fase 1: Multi-Kernel Self-Attention (MKSA)                     ║
    ║  Fase 2: Comparative Contextual Feature Fusion (CCFF)           ║  
    ║  Fase 3: Refinamiento Final                                     ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Verificar prerequisitos
    if not check_prerequisites():
        print("❌ Faltan prerequisitos. Abortando entrenamiento.")
        return
    
    start_total_time = time.time()
    
    # Configuración de fases
    phases = [
        {
            'name': 'Fase 1',
            'script': 'train.py',
            'description': 'EfficientNet + Multi-Kernel Self-Attention (MKSA)',
            'expected_output': 'best_ciff_net_phase1.pth'
        },
        {
            'name': 'Fase 2', 
            'script': 'train_phase2.py',
            'description': 'Comparative Contextual Feature Fusion (CCFF)',
            'expected_output': 'best_ciff_net_phase2.pth'
        },
        {
            'name': 'Fase 3',
            'script': 'train_phase3.py', 
            'description': 'Refinamiento Final con Fusión de Predicciones',
            'expected_output': 'best_ciff_net_phase3.pth'
        }
    ]
    
    completed_phases = []
    
    # Ejecutar fases secuencialmente
    for i, phase in enumerate(phases, 1):
        print(f"\n📋 Progreso: {i-1}/{len(phases)} fases completadas")
        
        # Verificar si la fase anterior generó su output (excepto para Fase 1)
        if i > 1:
            prev_output = phases[i-2]['expected_output']
            if not os.path.exists(prev_output):
                print(f"❌ Salida de fase anterior no encontrada: {prev_output}")
                print("💡 No se puede continuar sin completar la fase anterior")
                break
        
        # Ejecutar fase actual
        success = run_phase(phase['name'], phase['script'], phase['description'])
        
        if success:
            completed_phases.append(phase['name'])
            
            # Verificar que se generó el output esperado
            if os.path.exists(phase['expected_output']):
                print(f"✅ Modelo guardado: {phase['expected_output']}")
            else:
                print(f"⚠️  Advertencia: No se encontró {phase['expected_output']}")
        else:
            print(f"❌ {phase['name']} falló. Deteniendo entrenamiento.")
            break
    
    # Resumen final
    total_time = time.time() - start_total_time
    
    print(f"\n{'='*80}")
    print("📊 RESUMEN DEL ENTRENAMIENTO CIFF-NET")
    print(f"{'='*80}")
    
    print(f"⏱️  Tiempo total: {total_time/3600:.2f} horas")
    print(f"✅ Fases completadas: {len(completed_phases)}/{len(phases)}")
    
    for i, phase_name in enumerate(completed_phases):
        print(f"  {i+1}. {phase_name} ✅")
    
    if len(completed_phases) == len(phases):
        print("\n🎉 ¡ENTRENAMIENTO COMPLETO EXITOSO!")
        print("\n📁 Archivos generados:")
        
        # Listar archivos generados
        output_files = [
            'best_ciff_net_phase1.pth',
            'best_ciff_net_phase2.pth', 
            'best_ciff_net_phase3.pth',
            'training_history_ciff_phase1.png',
            'training_history_phase2.png',
            'ciff_net_complete_history.png',
            'confusion_matrix_ciff_phase1.png',
            'confusion_matrix_phase2.png',
            'confusion_matrix_complete.png'
        ]
        
        for file in output_files:
            if os.path.exists(file):
                print(f"  ✅ {file}")
            else:
                print(f"  ❌ {file} (no encontrado)")
        
        print(f"\n🏆 CIFF-Net entrenado completo y listo para uso!")
        print(f"🔧 Para hacer inferencia, usar: best_ciff_net_phase3.pth")
        
    else:
        print(f"\n⚠️  Entrenamiento incompleto: {len(completed_phases)}/{len(phases)} fases")
        print("💡 Revisa los errores arriba y reintenta las fases faltantes")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()