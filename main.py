from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import io
import base64
from typing import Optional, Dict, List
import logging
import traceback
from contextlib import asynccontextmanager

# Importar tus módulos
from phase1_feature_extraction import create_phase1_extractor
from phase2_cliff_detection_complete import create_phase2_complete_detector
from phase3_classification_complete import create_phase3_complete_classifier

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# MODELO COMPLETO CiffNet-ADC
# ================================

class CiffNetADCComplete(nn.Module):
    """
    Modelo completo CiffNet-ADC (3 fases) para deployment
    """
    
    def __init__(self, num_classes=7, cliff_threshold=0.15):
        super(CiffNetADCComplete, self).__init__()
        
        self.num_classes = num_classes
        self.cliff_threshold = cliff_threshold
        
        # Crear las 3 fases
        self.phase1 = create_phase1_extractor('efficientnet_b1', True)
        self.phase2 = create_phase2_complete_detector(256, cliff_threshold, num_classes)
        self.phase3 = create_phase3_complete_classifier(256, num_classes, cliff_threshold)
        
        # Metadatos para interpretación clínica
        self.class_names = [
            'Melanoma', 'Nevus', 'Basal Cell Carcinoma', 
            'Actinic Keratosis', 'Benign Keratosis', 
            'Dermatofibroma', 'Vascular Lesion'
        ]
        
        # Risk levels para cada clase
        self.risk_levels = {
            'Melanoma': 'HIGH', 'Basal Cell Carcinoma': 'HIGH',
            'Actinic Keratosis': 'MEDIUM', 'Nevus': 'LOW',
            'Benign Keratosis': 'LOW', 'Dermatofibroma': 'LOW',
            'Vascular Lesion': 'LOW'
        }
    
    def forward(self, x):
        """Forward pass completo a través de las 3 fases"""
        
        # FASE 1: Feature Extraction
        phase1_outputs = self.phase1(x)
        
        # FASE 2: Cliff Detection & Enhancement
        phase2_outputs = self.phase2(phase1_outputs['fused_features'])
        
        # FASE 3: Cliff-Aware Classification
        phase3_outputs = self.phase3(phase2_outputs, return_all=True)
        
        return {
            'phase1': phase1_outputs,
            'phase2': phase2_outputs,
            'phase3': phase3_outputs,
            'final_prediction': phase3_outputs['predictions'],
            'confidence': phase3_outputs['confidence'],
            'uncertainty': phase3_outputs['uncertainty']
        }

# ================================
# PYDANTIC MODELS
# ================================

class DiagnosisResult(BaseModel):
    """Resultado del diagnóstico"""
    predicted_class: str
    predicted_class_id: int
    confidence: float
    uncertainty: float
    risk_level: str
    is_cliff_case: bool
    
    # Probabilidades por clase
    class_probabilities: Dict[str, float]
    
    # Métricas técnicas
    cliff_score: float
    uncertainty_epistemic: float
    uncertainty_aleatoric: float
    
    # Recomendaciones clínicas
    clinical_recommendation: str
    confidence_level: str

class PhaseAnalysis(BaseModel):
    """Análisis detallado por fase"""
    phase1_features: Dict[str, float]
    phase2_cliff_analysis: Dict[str, float]
    phase3_classification: Dict[str, float]

class ComprehensiveResponse(BaseModel):
    """Respuesta completa del API"""
    diagnosis: DiagnosisResult
    phase_analysis: Optional[PhaseAnalysis] = None
    processing_time: float
    model_version: str = "CiffNet-ADC-v1.0"

# ================================
# GESTIÓN DEL MODELO GLOBAL
# ================================

# Global model instance
model_instance = None
device = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida del modelo"""
    global model_instance, device
    
    # Startup: Cargar modelo
    logger.info("🚀 Iniciando CiffNet-ADC Backend...")
    
    try:
        # Detectar device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"📱 Device detectado: {device}")
        
        # Crear modelo
        model_instance = CiffNetADCComplete(num_classes=7, cliff_threshold=0.15)
        
        # Cargar weights entrenados
        model_path = "results/models/ciffnet_epoch_100.pth"  # Tu modelo .pth
        
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
        
        # Cargar state dict
        if 'model_state_dict' in checkpoint:
            model_instance.load_state_dict(checkpoint['model_state_dict'])
        else:
            model_instance.load_state_dict(checkpoint)
        
        # Mover a device y set eval mode
        model_instance = model_instance.to(device)
        model_instance.eval()
        
        logger.info("✅ Modelo CiffNet-ADC cargado exitosamente")
        logger.info(f"📊 Parámetros totales: {sum(p.numel() for p in model_instance.parameters()):,}")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
    # Shutdown
    logger.info("🔄 Cerrando CiffNet-ADC Backend...")

# ================================
# FASTAPI APP
# ================================

app = FastAPI(
    title="CiffNet-ADC Dermatological Diagnosis API",
    description="Advanced cliff-aware neural network for skin lesion classification",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# UTILIDADES DE PROCESAMIENTO
# ================================

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocessar imagen para CiffNet-ADC
    """
    # Resize to 224x224 (EfficientNet-B1 input size)
    image = image.resize((224, 224), Image.Resampling.LANCZOS)
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Normalization (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img_array = (img_array - mean) / std
    
    # Convert to tensor [C, H, W]
    img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1))
    
    # Add batch dimension [1, C, H, W]
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor

def generate_clinical_recommendation(
    predicted_class: str, 
    confidence: float, 
    uncertainty: float, 
    is_cliff_case: bool
) -> str:
    """
    Generar recomendación clínica basada en resultados
    """
    risk_level = model_instance.risk_levels[predicted_class]
    
    if uncertainty > 0.7 or is_cliff_case:
        return "URGENT: High uncertainty detected. Recommend immediate dermatologist consultation and possible biopsy."
    
    elif risk_level == 'HIGH':
        if confidence > 0.8:
            return f"HIGH PRIORITY: {predicted_class} detected with high confidence. Recommend urgent dermatologist referral."
        else:
            return f"HIGH PRIORITY: Possible {predicted_class}. Recommend immediate dermatologist consultation for definitive diagnosis."
    
    elif risk_level == 'MEDIUM':
        if confidence > 0.7:
            return f"MEDIUM PRIORITY: {predicted_class} detected. Recommend dermatologist consultation within 2-4 weeks."
        else:
            return "MEDIUM PRIORITY: Uncertain diagnosis. Recommend dermatologist consultation for evaluation."
    
    else:  # LOW risk
        if confidence > 0.8:
            return f"LOW PRIORITY: {predicted_class} detected. Monitor for changes, routine follow-up recommended."
        else:
            return "LOW PRIORITY: Benign lesion likely. Monitor for changes, consult if concerned."

def get_confidence_level(confidence: float, uncertainty: float) -> str:
    """Clasificar nivel de confianza"""
    if uncertainty > 0.7:
        return "VERY_LOW"
    elif confidence > 0.9 and uncertainty < 0.3:
        return "VERY_HIGH"
    elif confidence > 0.8 and uncertainty < 0.4:
        return "HIGH"
    elif confidence > 0.6 and uncertainty < 0.6:
        return "MEDIUM"
    else:
        return "LOW"

# ================================
# ENDPOINTS
# ================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "CiffNet-ADC Dermatological Diagnosis API",
        "version": "1.0.0",
        "status": "operational",
        "model_loaded": model_instance is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    global model_instance, device
    
    return {
        "status": "healthy" if model_instance is not None else "unhealthy",
        "model_loaded": model_instance is not None,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }

@app.post("/diagnose", response_model=ComprehensiveResponse)
async def diagnose_lesion(
    file: UploadFile = File(...),
    include_phase_analysis: bool = False
):
    """
    Endpoint principal para diagnóstico de lesiones cutáneas
    
    Args:
        file: Imagen de la lesión (JPG, PNG, etc.)
        include_phase_analysis: Si incluir análisis detallado por fase
    
    Returns:
        ComprehensiveResponse: Diagnóstico completo con recomendaciones
    """
    import time
    start_time = time.time()
    
    try:
        # Validar que el modelo esté cargado
        if model_instance is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Validar tipo de archivo
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Leer y procesar imagen
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocessar
        input_tensor = preprocess_image(image).to(device)
        
        # Inferencia
        with torch.no_grad():
            outputs = model_instance(input_tensor)
        
        # Extraer resultados
        phase3_out = outputs['phase3']
        phase2_out = outputs['phase2']
        
        # Convertir a CPU y extraer valores
        predictions = phase3_out['predictions'].cpu().numpy()[0]
        probabilities = phase3_out['probabilities'].cpu().numpy()[0]
        confidence = float(phase3_out['confidence'].cpu().numpy()[0])
        uncertainty = float(phase3_out['uncertainty'].cpu().numpy()[0])
        
        # Cliff analysis
        cliff_score = float(phase2_out['cliff_score'].cpu().numpy()[0])
        is_cliff_case = bool(phase2_out['cliff_mask'].cpu().numpy()[0])
        
        # Uncertainty breakdown
        epistemic_unc = float(phase2_out['epistemic_uncertainty'].cpu().numpy()[0].mean())
        aleatoric_unc = float(phase2_out['aleatoric_uncertainty'].cpu().numpy()[0])
        
        # Mapear a nombres de clases
        predicted_class = model_instance.class_names[predictions]
        class_probabilities = {
            name: float(prob) 
            for name, prob in zip(model_instance.class_names, probabilities)
        }
        
        # Generar recomendaciones
        clinical_recommendation = generate_clinical_recommendation(
            predicted_class, confidence, uncertainty, is_cliff_case
        )
        confidence_level = get_confidence_level(confidence, uncertainty)
        
        # Crear resultado del diagnóstico
        diagnosis = DiagnosisResult(
            predicted_class=predicted_class,
            predicted_class_id=int(predictions),
            confidence=confidence,
            uncertainty=uncertainty,
            risk_level=model_instance.risk_levels[predicted_class],
            is_cliff_case=is_cliff_case,
            class_probabilities=class_probabilities,
            cliff_score=cliff_score,
            uncertainty_epistemic=epistemic_unc,
            uncertainty_aleatoric=aleatoric_unc,
            clinical_recommendation=clinical_recommendation,
            confidence_level=confidence_level
        )
        
        # Análisis por fase (opcional)
        phase_analysis = None
        if include_phase_analysis:
            phase1_out = outputs['phase1']
            
            phase_analysis = PhaseAnalysis(
                phase1_features={
                    "backbone_features_mean": float(phase1_out['fused_features'].mean()),
                    "feature_diversity": float(torch.std(phase1_out['fused_features'])),
                    "multi_scale_activation": float(phase1_out['multi_scale_features'].mean())
                },
                phase2_cliff_analysis={
                    "cliff_score": cliff_score,
                    "spatial_cliff": float(phase2_out['spatial_cliff'].cpu().numpy()[0]),
                    "multiscale_cliff": float(phase2_out['multiscale_cliff'].cpu().numpy()[0]),
                    "epistemic_uncertainty": epistemic_unc,
                    "aleatoric_uncertainty": aleatoric_unc,
                    "attention_entropy": phase2_out['analysis']['attention_entropy']
                },
                phase3_classification={
                    "main_classifier_confidence": float(torch.max(torch.softmax(phase3_out['main_logits'], dim=1))),
                    "cliff_classifier_confidence": float(torch.max(torch.softmax(phase3_out['cliff_logits'], dim=1))),
                    "classifier_used": "cliff" if is_cliff_case else "main",
                    "monte_carlo_uncertainty": uncertainty
                }
            )
        
        # Tiempo de procesamiento
        processing_time = time.time() - start_time
        
        # Respuesta completa
        response = ComprehensiveResponse(
            diagnosis=diagnosis,
            phase_analysis=phase_analysis,
            processing_time=processing_time
        )
        
        logger.info(f"✅ Diagnóstico completado: {predicted_class} (conf: {confidence:.3f}, unc: {uncertainty:.3f})")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error en diagnóstico: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Diagnosis error: {str(e)}")

@app.post("/batch_diagnose")
async def batch_diagnose(files: List[UploadFile] = File(...)):
    """
    Diagnóstico por lotes (múltiples imágenes)
    """
    if len(files) > 10:  # Limitar batch size
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    
    for i, file in enumerate(files):
        try:
            # Usar el endpoint individual
            result = await diagnose_lesion(file, include_phase_analysis=False)
            results.append({
                "image_index": i,
                "filename": file.filename,
                "diagnosis": result.diagnosis,
                "processing_time": result.processing_time
            })
        except Exception as e:
            results.append({
                "image_index": i,
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"batch_results": results}

@app.get("/model_info")
async def get_model_info():
    """Información detallada del modelo"""
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "CiffNet-ADC",
        "version": "1.0.0",
        "architecture": "3-Phase Cliff-Aware Dermatological Classifier",
        "classes": model_instance.class_names,
        "num_classes": model_instance.num_classes,
        "cliff_threshold": model_instance.cliff_threshold,
        "input_size": "224x224x3",
        "phases": {
            "phase1": "Feature Extraction (EfficientNet-B1)",
            "phase2": "Cliff Detection & Enhancement (CFM+CRI+CAFE)",
            "phase3": "Adaptive Dual Classification"
        },
        "capabilities": [
            "Uncertainty Quantification",
            "Cliff-aware Classification", 
            "Clinical Risk Assessment",
            "Monte Carlo Dropout",
            "Multi-loss Training"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False,  # False en producción
        workers=1  # Single worker para GPU sharing
    )