# EPP Cafetería UAO — Sistema de Monitoreo de Higiene y Bioseguridad

**Procesamiento Digital de Imágenes · Prof. Nicolas Llanos Neuta · UAO 2026**

## Descripción

Sistema de visión computacional en dos etapas:
1. **YOLOv11n** (preentrenado COCO) — detecta personas en la imagen
2. **CNN MobileNetV2** (entrenado con dataset propio) — clasifica EPP por persona

## Arquitectura del pipeline

```
Imagen entrada
    │
    ▼ Preprocesamiento (resize 640, normalización ImageNet)
    │
    ▼ YOLOv11n → bounding boxes de personas
    │
    ├──► Recorte persona 1 ──► MobileNetV2 CNN ──► [delantal✓, gorro✓, guantes✗, tapabocas✓]
    ├──► Recorte persona 2 ──► MobileNetV2 CNN ──► [delantal✓, gorro✓, guantes✓, tapabocas✓]
    └──► ...
    │
    ▼ Acumulador → estadísticas del turno → Dashboard
```

## Dataset

```
dataset/
├── train/  delantal(147) gorro(174) guantes(153) tapabocas(180)
├── valid/  delantal(20)  gorro(25)  guantes(28)  tapabocas(40)
└── test/   delantal(21)  gorro(14)  guantes(23)  tapabocas(18)
```

## Estructura del proyecto

```
epp-cafeteria/
├── backend/
│   ├── train.py          ← Pipeline de entrenamiento CNN (ver aquí el código PDI)
│   ├── main.py           ← API FastAPI (YOLOv11 + CNN en cascada)
│   ├── model/
│   │   └── epp_classifier.pt   ← modelo entrenado
│   └── requirements.txt
├── frontend/
│   ├── app/page.tsx      ← Dashboard Next.js
│   ├── components/
│   └── lib/
├── dataset/
│   ├── train/
│   ├── valid/
│   └── test/
└── README.md
```

## Instalación y uso

### Backend

```bash
cd backend

# Crear entorno virtual
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

# Instalar dependencias
pip install -r requirements.txt

# 1. Entrenar el modelo (necesario antes de correr el servidor)
python train.py --dataset ../dataset --epochs 30 --ft_epochs 10 --batch 32

# 2. Correr el servidor
uvicorn main:app --reload --port 8000
```

El backend queda en: `http://localhost:8000`
Documentación automática: `http://localhost:8000/docs`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

La app queda en: `http://localhost:3000`

## Deploy en producción

### Backend → Railway

1. Ve a [railway.app](https://railway.app) → New Project → Deploy from GitHub
2. Selecciona la carpeta `backend/`
3. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. El modelo `model/epp_classifier.pt` debe estar en el repo

### Frontend → Vercel

1. Ve a [vercel.com](https://vercel.com) → New Project → importa el repo
2. Root directory: `frontend/`
3. Environment variable: `NEXT_PUBLIC_BACKEND_URL=https://tu-backend.railway.app`
4. Deploy

## Uso del sistema

### Modo manual
- Clic en "↑ Subir imagen" → se analiza con el pipeline completo
- Muestra bounding box por persona y EPP detectados/faltantes

### Modo simulación
- **Demo (12s):** una inspección cada 12s — para la presentación
- **Real (5 min):** simula el scheduler real (1 foto cada 5 min)
- El acumulador registra todas las inspecciones del turno

### Reporte
- "↓ Descargar reporte del turno" genera un `.txt` con el resumen completo
