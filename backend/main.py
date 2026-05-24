"""
=============================================================================
  SISTEMA EPP CAFETERÍA UAO — Backend FastAPI
=============================================================================

Pipeline de inferencia:
  1. Recibe imagen (upload o base64)
  2. Preprocesamiento: resize + normalización
  3. YOLOv11n (COCO) → detecta personas → bounding boxes
  4. Por cada persona: recorte → CNN MobileNetV2 → detecta EPP presentes
  5. Retorna JSON con personas, EPP detectados/faltantes, compliance %

Uso:
    uvicorn main:app --reload --port 8000
=============================================================================
"""

import io
import json
import base64
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel


# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
EPP_CLASSES  = ['delantal', 'gorro', 'guantes', 'tapabocas']
IMG_SIZE     = 224
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH   = Path(__file__).parent / 'model' / 'epp_classifier.pt'
CONF_PERSON  = 0.45   # confianza mínima para detectar persona

# Umbral por clase: tapabocas tiene umbral más bajo porque el modelo
# fue entrenado con imágenes aisladas y no generaliza bien en fotos reales
CONF_EPP_PER_CLASS = {
    'delantal':  0.40,   # activa en ropa colorida; exige alta certeza
    'gorro':     0.33,   # rango real 0.40-0.65; solo detecta cuando el modelo es muy seguro
    'guantes':   0.30,   # rango real 0.38-0.80
    'tapabocas': 0.25,   # señal siempre baja; umbral permisivo
}

EPP_COLORS = {
    'delantal':  (77,  171, 247),
    'gorro':     (0,   229, 176),
    'guantes':   (255, 179,  71),
    'tapabocas': (192, 132, 252),
}
VIOLATION_COLOR = (255, 77, 77)
PERSON_COLOR    = (100, 100, 255)


# ─────────────────────────────────────────────────────────────
#  PREPROCESAMIENTO (igual que en train.py)
# ─────────────────────────────────────────────────────────────
PREPROCESS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────
#  MODELO CNN (misma arquitectura que train.py)
# ─────────────────────────────────────────────────────────────
class EPPClassifier(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        backbone = models.mobilenet_v2(weights=None)
        self.features   = backbone.features
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────
#  CARGA DE MODELOS AL INICIO
# ─────────────────────────────────────────────────────────────
print(f"\n[EPP Backend] Dispositivo: {DEVICE.upper()}")
print("[EPP Backend] Cargando YOLOv11n (detección de personas)...")
yolo_model = YOLO('yolo11n.pt')   # descarga automática si no existe

print("[EPP Backend] Cargando CNN clasificador EPP...")
epp_model = EPPClassifier(num_classes=len(EPP_CLASSES))
if MODEL_PATH.exists():
    epp_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    print(f"[EPP Backend] ✓ Modelo cargado desde {MODEL_PATH}")
else:
    print(f"[EPP Backend] ⚠ Modelo no encontrado en {MODEL_PATH}")
    print("[EPP Backend]   Ejecuta: python train.py primero")
epp_model.eval()
epp_model.to(DEVICE)


# ─────────────────────────────────────────────────────────────
#  FASTAPI APP
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="EPP Cafetería UAO",
    description="Sistema de monitoreo de EPP con YOLOv11 + CNN MobileNetV2",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
#  LÓGICA DE INFERENCIA
# ─────────────────────────────────────────────────────────────
def preprocess_image(img: Image.Image) -> Image.Image:
    """Convierte a RGB y normaliza el tamaño para el pipeline."""
    return img.convert('RGB')


def detect_persons(img: Image.Image) -> list[dict]:
    """
    Paso 1: YOLOv11n detecta personas (clase 0 en COCO).
    Retorna lista de bounding boxes [x1,y1,x2,y2] con confianza.
    """
    results = yolo_model(img, classes=[0], conf=CONF_PERSON, verbose=False)
    persons = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            # Expandir box un 10% para incluir contexto
            w, h = img.size
            pad_x = int((x2 - x1) * 0.05)
            pad_y = int((y2 - y1) * 0.05)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            persons.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'conf': conf})
    return persons


@torch.no_grad()
def classify_epp(crop: Image.Image) -> dict[str, float]:
    """CNN MobileNetV2: clasifica EPP en un recorte. Retorna probabilidades sigmoid."""
    tensor = PREPROCESS(crop).unsqueeze(0).to(DEVICE)
    logits = epp_model(tensor)[0]
    probs  = torch.sigmoid(logits)
    return {cls: float(probs[i]) for i, cls in enumerate(EPP_CLASSES)}


def classify_epp_zoned(person_crop: Image.Image) -> dict[str, float]:
    """
    Clasificación con media geométrica entre crop completo y crop de zona.
    La media geométrica requiere que AMBOS crops coincidan — reduce falsos positivos
    causados por tomar el máximo (que dispara con cualquier crop espurio).
    """
    w, h = person_crop.size
    head_crop  = person_crop.crop((0, 0,           w, max(1, int(h * 0.42))))
    torso_crop = person_crop.crop((0, int(h*0.18), w, max(1, int(h * 0.76))))
    hands_crop = person_crop.crop((0, int(h*0.50), w, h))

    full  = classify_epp(person_crop)
    head  = classify_epp(head_crop)
    torso = classify_epp(torso_crop)
    hands = classify_epp(hands_crop)

    def geo(a: float, b: float) -> float:
        return (a * b) ** 0.5

    return {
        'delantal':  geo(full['delantal'],  torso['delantal']),
        'gorro':     geo(full['gorro'],     head['gorro']),
        'guantes':   geo(full['guantes'],   hands['guantes']),
        'tapabocas': geo(full['tapabocas'], head['tapabocas']),
    }


def _load_fonts(size_sm: int, size_md: int):
    """Carga fuente TrueType; cae a la fuente por defecto si no está disponible."""
    for path in [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size_sm), ImageFont.truetype(path, size_md)
        except OSError:
            continue
    fallback = ImageFont.load_default()
    return fallback, fallback


def draw_results(img: Image.Image, persons: list[dict]) -> Image.Image:
    """
    Dibuja bboxes de personas con estado visible.
    El detalle de cada EPP se muestra en las tarjetas del dashboard, no aquí,
    para mantener la imagen legible sin importar qué tan pequeño sea el bbox.
    """
    draw = ImageDraw.Draw(img, 'RGBA')

    # Escala proporcional al tamaño real de la imagen
    scale  = max(1.5, img.width / 640)
    _, font_hd = _load_fonts(int(14 * scale), int(18 * scale))
    lw = max(2, int(3 * scale))
    cs = int(22 * scale)   # longitud de las esquinas decorativas

    for i, p in enumerate(persons):
        x1, y1, x2, y2 = p['bbox']
        has_violation = len(p['violations']) > 0
        compliance    = p['compliance']
        border_color  = VIOLATION_COLOR if has_violation else (0, 229, 176)
        status_text   = 'INCUMPLE' if has_violation else 'CUMPLE'

        # ── Bounding box ──────────────────────────────────────
        draw.rectangle([x1, y1, x2, y2], outline=border_color + (210,), width=lw)

        # ── Esquinas decorativas ──────────────────────────────
        ew = lw + 2
        draw.line([(x1, y1 + cs), (x1, y1), (x1 + cs, y1)], fill=border_color + (255,), width=ew)
        draw.line([(x2 - cs, y1), (x2, y1), (x2, y1 + cs)], fill=border_color + (255,), width=ew)
        draw.line([(x1, y2 - cs), (x1, y2), (x1 + cs, y2)], fill=border_color + (255,), width=ew)
        draw.line([(x2 - cs, y2), (x2, y2), (x2, y2 - cs)], fill=border_color + (255,), width=ew)

        # ── Etiqueta: "P1  50%  CUMPLE" encima del bbox ───────
        pad      = int(8 * scale)
        label    = f"P{i + 1}   {compliance}%   {status_text}"
        hdr_h    = int(28 * scale)
        hdr_top  = max(0, y1 - hdr_h - 2)
        hdr_bot  = max(hdr_top + 1, y1 - 2)   # garantiza hdr_bot > hdr_top
        draw.rectangle([x1, hdr_top, x2, hdr_bot], fill=(15, 15, 35, 225))
        draw.text(
            (x1 + pad, hdr_top + (hdr_h - int(18 * scale)) // 2),
            label,
            fill=border_color + (255,),
            font=font_hd,
        )

        # ── Mini-indicadores de EPP (puntos de color) ─────────
        # Calcula tamaño de punto para que los 4 quepan dentro del bbox
        available_w = (x2 - x1) - 2 * pad
        n       = len(EPP_CLASSES)
        dot_d   = min(int(20 * scale), available_w // (n * 2))   # diámetro
        dot_gap = max(2, (available_w - n * dot_d) // (n + 1))
        dot_x   = x1 + pad + dot_gap
        dot_y   = y2 - dot_d - int(6 * scale)
        for epp in EPP_CLASSES:
            present   = epp in p['detected']
            color     = EPP_COLORS.get(epp, (200, 200, 200))
            fill_c    = color + (210,) if present else VIOLATION_COLOR + (130,)
            outline_c = color + (255,) if present else VIOLATION_COLOR + (255,)
            draw.ellipse(
                [dot_x, dot_y, dot_x + dot_d, dot_y + dot_d],
                fill=fill_c, outline=outline_c, width=max(1, lw - 1)
            )
            dot_x += dot_d + dot_gap

    return img


def run_pipeline(img: Image.Image) -> dict:
    """Pipeline completo: preprocesamiento → personas → EPP por persona → resultado."""
    img = preprocess_image(img)
    persons_raw = detect_persons(img)

    if not persons_raw:
        # Sin persona detectada: analizar imagen completa con zonas
        probs = classify_epp_zoned(img)
        detected   = [c for c, p in probs.items() if p >= CONF_EPP_PER_CLASS[c]]
        violations = [c for c in EPP_CLASSES if c not in detected]
        compliance = round(len(detected) / len(EPP_CLASSES) * 100)
        persons_out = [{
            'id': 1, 'bbox': [0, 0, img.width, img.height],
            'conf': 1.0, 'detected': detected, 'violations': violations,
            'probs': probs, 'compliance': compliance
        }]
    else:
        persons_out = []
        for i, p in enumerate(persons_raw):
            crop  = img.crop((p['x1'], p['y1'], p['x2'], p['y2']))
            probs = classify_epp_zoned(crop)   # usa zonas para mejor precisión
            detected   = [c for c, pr in probs.items() if pr >= CONF_EPP_PER_CLASS[c]]
            violations = [c for c in EPP_CLASSES if c not in detected]
            compliance = round(len(detected) / len(EPP_CLASSES) * 100)
            persons_out.append({
                'id': i + 1,
                'bbox': [p['x1'], p['y1'], p['x2'], p['y2']],
                'conf': round(p['conf'], 3),
                'detected': detected,
                'violations': violations,
                'probs': {k: round(v, 3) for k, v in probs.items()},
                'compliance': compliance,
            })

    # Imagen anotada
    img_draw = img.copy()
    img_draw = draw_results(img_draw, persons_out)
    buf = io.BytesIO()
    img_draw.save(buf, format='JPEG', quality=90)
    annotated_b64 = base64.b64encode(buf.getvalue()).decode()

    global_compliance = round(
        sum(p['compliance'] for p in persons_out) / len(persons_out)
    )

    return {
        'persons': persons_out,
        'global_compliance': global_compliance,
        'total_persons': len(persons_out),
        'annotated_image': annotated_b64,
    }


# ─────────────────────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "system": "EPP Cafetería UAO",
        "model_loaded": MODEL_PATH.exists(),
        "device": DEVICE,
        "classes": EPP_CLASSES,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Recibe una imagen, corre el pipeline completo y retorna:
    - persons: lista de personas con EPP detectados/faltantes
    - global_compliance: % cumplimiento global
    - annotated_image: imagen con bounding boxes en base64
    """
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Modelo CNN no entrenado. Ejecuta: python train.py"
        )
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        result = run_pipeline(img)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class Base64Request(BaseModel):
    image: str   # base64 sin prefijo data:...

@app.post("/detect/base64")
async def detect_base64(req: Base64Request):
    """Alternativa: recibe imagen en base64 (para el frontend Next.js)."""
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503,
                            detail="Modelo no entrenado. Ejecuta: python train.py")
    try:
        img_bytes = base64.b64decode(req.image)
        img = Image.open(io.BytesIO(img_bytes))
        result = run_pipeline(img)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
