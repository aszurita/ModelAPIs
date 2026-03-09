"""
BananaVision API — main.py
==========================
Instalar:  pip install fastapi uvicorn python-multipart pillow numpy tensorflow
Correr:    uvicorn main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /                  → health check
    POST /enfermedades      → Modelo A solo
    POST /nutrientes        → Modelo B solo
    POST /diagnostico       → Pipeline completo (A → si sana → B)
"""

import io
import os
import time

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACION
# ──────────────────────────────────────────────────────────────────────────────
RUTA_ENF = os.getenv("RUTA_MODELO_ENF", "modelos/enfermedades.keras")
RUTA_NUT = os.getenv("RUTA_MODELO_NUT", "modelos/nutrientes.keras")
IMG_SIZE  = (224, 224)
UMBRAL    = 0.45   # confianza minima; debajo de esto → "incierto"

CLASES_ENF = ["Cordana", "Fusarium", "Sanas", "SigatokaNegra"]
CLASES_NUT = ["Boron", "Calcium", "Iron", "Magnesium",
              "Manganese", "Potassium", "Sano", "Sulphur"]

# Informacion agronomica por clase
INFO_ENF = {
    "Cordana": {
        "urgencia":    "MEDIA",
        "accion":      "Deshoje sanitario. Aplicar fungicida cuprico preventivo.",
        "exportable":  "SI (con monitoreo activo)",
        "recuperable": "SI",
    },
    "Fusarium": {
        "urgencia":    "CRITICA",
        "accion":      "CUARENTENA TOTAL. Aislar lote. Llamar a AGROCALIDAD: 1800-AGRO.",
        "exportable":  "NO — suelo inutilizable 30-40 anos",
        "recuperable": "NO",
    },
    "Sanas": {
        "urgencia":    "BAJA",
        "accion":      "Planta sana. Continuar monitoreo semanal.",
        "exportable":  "SI",
        "recuperable": "N/A",
    },
    "SigatokaNegra": {
        "urgencia":    "MEDIA-ALTA",
        "accion":      "Deshoje urgente. Fungicida sistemico antes de la proxima lluvia.",
        "exportable":  "SI (riesgo maduracion prematura)",
        "recuperable": "SI (requiere control permanente)",
    },
}

INFO_NUT = {
    "Boron": {
        "sintoma":       "Hojas jovenes deformadas, puntas necroticas",
        "tratamiento":   "Borax foliar 1-2 g/L. Aplicar en horas frescas.",
        "producto_tipo": "Fertilizante foliar — Boro",
    },
    "Calcium": {
        "sintoma":       "Bordes foliares secos, necrosis marginal",
        "tratamiento":   "Nitrato de calcio foliar 0.5%",
        "producto_tipo": "Fertilizante foliar — Calcio",
    },
    "Iron": {
        "sintoma":       "Clorosis internervial en hojas jovenes",
        "tratamiento":   "Quelato de hierro foliar (verificar pH < 6.5)",
        "producto_tipo": "Quelato de hierro",
    },
    "Magnesium": {
        "sintoma":       "Clorosis entre nervaduras en hojas viejas",
        "tratamiento":   "Sulfato de magnesio MgSO4 foliar 1%",
        "producto_tipo": "Fertilizante — Magnesio",
    },
    "Manganese": {
        "sintoma":       "Manchas cloroticas irregulares, aspecto moteado",
        "tratamiento":   "Sulfato de manganeso foliar en horas frescas",
        "producto_tipo": "Fertilizante foliar — Manganeso",
    },
    "Potassium": {
        "sintoma":       "Bordes quemados, fruta pequena, hojas viejas afectadas",
        "tratamiento":   "KCl o K2SO4 al suelo segun analisis",
        "producto_tipo": "Fertilizante suelo — Potasio",
    },
    "Sano": {
        "sintoma":       "Planta en estado nutricional optimo",
        "tratamiento":   "Mantener plan de fertilizacion actual",
        "producto_tipo": "N/A",
    },
    "Sulphur": {
        "sintoma":       "Clorosis general en hojas jovenes",
        "tratamiento":   "Sulfato de amonio o azufre elemental al suelo",
        "producto_tipo": "Fertilizante — Azufre",
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# CARGAR MODELOS AL ARRANCAR (una sola vez en memoria)
# ──────────────────────────────────────────────────────────────────────────────
print("Cargando modelos TensorFlow...")
t0 = time.time()

if not os.path.exists(RUTA_ENF):
    raise FileNotFoundError(f"No encontre el modelo de enfermedades en: {RUTA_ENF}")
if not os.path.exists(RUTA_NUT):
    raise FileNotFoundError(f"No encontre el modelo de nutrientes en: {RUTA_NUT}")

modelo_enf = tf.keras.models.load_model(RUTA_ENF)
print(f"  Modelo A (Enfermedades) listo — {time.time()-t0:.1f}s")

t1 = time.time()
modelo_nut = tf.keras.models.load_model(RUTA_NUT)
print(f"  Modelo B (Nutrientes) listo   — {time.time()-t1:.1f}s")
print(f"Ambos modelos en memoria ({time.time()-t0:.1f}s total)\n")

# ──────────────────────────────────────────────────────────────────────────────
# FUNCIONES INTERNAS
# ──────────────────────────────────────────────────────────────────────────────
def bytes_a_tensor(imagen_bytes: bytes) -> np.ndarray:
    """
    Convierte bytes de imagen (jpg/png) al array que espera EfficientNetB3.
    Retorna shape (1, 224, 224, 3) listo para model.predict().
    """
    img = Image.open(io.BytesIO(imagen_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)        # escala para EfficientNet
    return np.expand_dims(arr, axis=0) # agregar dimension de batch


def predecir(modelo, tensor: np.ndarray, clases: list) -> dict:
    """
    Corre inferencia y retorna clase, confianza y probabilidades de todas las clases.
    """
    preds     = modelo.predict(tensor, verbose=0)[0]
    idx       = int(np.argmax(preds))
    confianza = float(preds[idx])

    return {
        "clase":             clases[idx],
        "confianza":         round(confianza * 100, 1),
        "confiable":         confianza >= UMBRAL,
        "todas_las_clases":  {c: round(float(p) * 100, 2) for c, p in zip(clases, preds)},
    }


def validar_imagen(archivo: UploadFile):
    if not archivo.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Debe ser una imagen jpg/png. Recibido: {archivo.content_type}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# APP
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="BananaVision API",
    description="Diagnostico de enfermedades y deficiencias nutricionales en banano",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# GET /  →  health check
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {
        "ok":      True,
        "version": "1.0.0",
        "modelos_cargados": {
            "enfermedades": modelo_enf is not None,
            "nutrientes":   modelo_nut is not None,
        },
        "clases": {
            "enfermedades": CLASES_ENF,
            "nutrientes":   CLASES_NUT,
        }
    }


# ──────────────────────────────────────────────────────────────────────────────
# POST /enfermedades  →  Modelo A solo
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/enfermedades")
async def detectar_enfermedad(foto: UploadFile = File(...)):
    """
    Manda foto de hoja → retorna si tiene enfermedad y que hacer.

    curl -X POST https://tu-api/enfermedades -F "foto=@hoja.jpg"

    Respuesta:
    {
        "clase": "Fusarium",
        "confianza": 97.3,
        "confiable": true,
        "info": {
            "urgencia": "CRITICA",
            "accion": "CUARENTENA TOTAL...",
            "exportable": "NO",
            "recuperable": "NO"
        },
        "todas_las_clases": {
            "Cordana": 0.5,
            "Fusarium": 97.3,
            "Sanas": 1.9,
            "SigatokaNegra": 0.3
        },
        "tiempo_ms": 210.4
    }
    """
    validar_imagen(foto)
    try:
        t0           = time.time()
        imagen_bytes = await foto.read()
        tensor       = bytes_a_tensor(imagen_bytes)
        resultado    = predecir(modelo_enf, tensor, CLASES_ENF)

        resultado["info"]      = INFO_ENF.get(resultado["clase"], {})
        resultado["tiempo_ms"] = round((time.time() - t0) * 1000, 1)

        if not resultado["confiable"]:
            resultado["aviso"] = (
                f"Confianza baja ({resultado['confianza']}%). "
                "Toma la foto con mejor luz, sin sombras, enfocada en la hoja."
            )
        return resultado

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# POST /nutrientes  →  Modelo B solo
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/nutrientes")
async def detectar_nutriente(foto: UploadFile = File(...)):
    """
    Manda foto de hoja → retorna la deficiencia nutricional y tratamiento.

    curl -X POST https://tu-api/nutrientes -F "foto=@hoja.jpg"

    Respuesta:
    {
        "clase": "Magnesium",
        "confianza": 84.2,
        "confiable": true,
        "info": {
            "sintoma": "Clorosis entre nervaduras en hojas viejas",
            "tratamiento": "Sulfato de magnesio MgSO4 foliar 1%",
            "producto_tipo": "Fertilizante — Magnesio"
        },
        "todas_las_clases": { "Boron": 1.2, "Calcium": 0.8, "Magnesium": 84.2, ... },
        "tiempo_ms": 198.7
    }
    """
    validar_imagen(foto)
    try:
        t0           = time.time()
        imagen_bytes = await foto.read()
        tensor       = bytes_a_tensor(imagen_bytes)
        resultado    = predecir(modelo_nut, tensor, CLASES_NUT)

        resultado["info"]      = INFO_NUT.get(resultado["clase"], {})
        resultado["tiempo_ms"] = round((time.time() - t0) * 1000, 1)

        if not resultado["confiable"]:
            resultado["aviso"] = (
                f"Confianza baja ({resultado['confianza']}%). "
                "Considera hacer un analisis de suelo para confirmar."
            )
        return resultado

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# POST /diagnostico  →  Pipeline completo (A → si sana → B)
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/diagnostico")
async def diagnostico_completo(foto: UploadFile = File(...)):
    """
    Endpoint principal. Corre los dos modelos en orden logico:
      1. Modelo A: hay enfermedad?
         - Si hay  → retorna resultado de enfermedad, no sigue
         - Si sana → pasa al Modelo B
      2. Modelo B: que deficiencia nutricional tiene?

    curl -X POST https://tu-api/diagnostico -F "foto=@hoja.jpg"

    Respuesta con enfermedad:
    {
        "flujo": "enfermedades",
        "clase": "Fusarium",
        "confianza": 97.3,
        "confiable": true,
        "info": { "urgencia": "CRITICA", "accion": "...", ... },
        "modelo_A": { ... resultado completo ... },
        "modelo_B": null,
        "tiempo_ms": 350.1
    }

    Respuesta sana con deficiencia:
    {
        "flujo": "enfermedades → nutrientes",
        "clase": "Magnesium",
        "confianza": 84.2,
        "confiable": true,
        "info": { "sintoma": "...", "tratamiento": "..." },
        "modelo_A": { ... },
        "modelo_B": { ... },
        "tiempo_ms": 520.3
    }

    Respuesta completamente sana:
    {
        "flujo": "enfermedades → nutrientes",
        "clase": "Sano",
        "confianza": 91.0,
        "confiable": true,
        "info": { "sintoma": "Planta sana", "tratamiento": "..." },
        "modelo_A": { ... },
        "modelo_B": { ... },
        "tiempo_ms": 510.8
    }
    """
    validar_imagen(foto)
    try:
        t0           = time.time()
        imagen_bytes = await foto.read()
        tensor       = bytes_a_tensor(imagen_bytes)

        # Paso 1: Modelo A
        res_enf         = predecir(modelo_enf, tensor, CLASES_ENF)
        res_enf["info"] = INFO_ENF.get(res_enf["clase"], {})

        if res_enf["clase"] != "Sanas":
            # Tiene enfermedad — no continuar al modelo de nutrientes
            return {
                "flujo":     "enfermedades",
                "clase":     res_enf["clase"],
                "confianza": res_enf["confianza"],
                "confiable": res_enf["confiable"],
                "info":      res_enf["info"],
                "modelo_A":  res_enf,
                "modelo_B":  None,
                "tiempo_ms": round((time.time() - t0) * 1000, 1),
            }

        # Paso 2: Modelo B (solo si esta sana)
        res_nut         = predecir(modelo_nut, tensor, CLASES_NUT)
        res_nut["info"] = INFO_NUT.get(res_nut["clase"], {})

        return {
            "flujo":     "enfermedades → nutrientes",
            "clase":     res_nut["clase"],
            "confianza": res_nut["confianza"],
            "confiable": res_nut["confiable"],
            "info":      res_nut["info"],
            "modelo_A":  res_enf,
            "modelo_B":  res_nut,
            "tiempo_ms": round((time.time() - t0) * 1000, 1),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# POST /diagnostico/completo  →  Modelos A + B + Google Earth Engine
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/diagnostico/completo")
async def diagnostico_con_gee(
    foto: UploadFile = File(...),
    lat: float = None,
    lon: float = None,
):
    """
    Pipeline completo: Modelo A + Modelo B + Google Earth Engine.

    curl -X POST https://tu-api/diagnostico/completo \\
         -F "foto=@hoja.jpg" \\
         -F "lat=-2.09" \\
         -F "lon=-79.54"

    Respuesta:
    {
        "flujo": "enfermedades → nutrientes + GEE",
        "clase": "Sano",
        "confianza": 91.0,
        "confiable": true,
        "info": { ... },
        "modelo_A": { ... },
        "modelo_B": { ... },
        "satelital": {
            "ndvi": 0.62,
            "precipitacion_mm": 142.3,
            "humedad_suelo_m3m3": 0.31,
            "temperatura_c": 27.4,
            "riesgo_inundacion": "MEDIO",
            "tipo_suelo_nombre": "Franco-arcilloso",
            "riesgo_suelo_fusarium": "ALTO",
            "indice_riesgo_ambiental": 6,
            "nivel_riesgo_fusarium": "ALTO"
        },
        "alerta_gee": {
            "nivel": "ALTO",
            "titulo": "Condiciones favorables para hongos",
            "mensaje": "Tu lote tiene condiciones de riesgo elevado...",
            "accion": "Aplicar fungicida preventivo."
        },
        "advertencia_cruzada": "Las condiciones del suelo son ALTO riesgo para Fusarium aunque la foto no muestre sintomas.",
        "tiempo_ms": 2840.1
    }

    Si no mandas lat/lon el campo satelital explica que faltan coordenadas
    y el endpoint funciona igual que /diagnostico normal.
    """
    validar_imagen(foto)
    try:
        t0           = time.time()
        imagen_bytes = await foto.read()
        tensor       = bytes_a_tensor(imagen_bytes)

        # ── Paso 1: Modelo A ─────────────────────────────────────────────────
        res_enf         = predecir(modelo_enf, tensor, CLASES_ENF)
        res_enf["info"] = INFO_ENF.get(res_enf["clase"], {})

        if res_enf["clase"] != "Sanas":
            res_modelos = {
                "flujo":     "enfermedades + GEE",
                "clase":     res_enf["clase"],
                "confianza": res_enf["confianza"],
                "confiable": res_enf["confiable"],
                "info":      res_enf["info"],
                "modelo_A":  res_enf,
                "modelo_B":  None,
            }
        else:
            # ── Paso 2: Modelo B ─────────────────────────────────────────────
            res_nut         = predecir(modelo_nut, tensor, CLASES_NUT)
            res_nut["info"] = INFO_NUT.get(res_nut["clase"], {})
            res_modelos = {
                "flujo":     "enfermedades → nutrientes + GEE",
                "clase":     res_nut["clase"],
                "confianza": res_nut["confianza"],
                "confiable": res_nut["confiable"],
                "info":      res_nut["info"],
                "modelo_A":  res_enf,
                "modelo_B":  res_nut,
            }

        # ── Paso 3: Google Earth Engine ──────────────────────────────────────
        satelital   = None
        alerta_gee  = None
        advertencia = None

        if lat is not None and lon is not None:
            try:
                from gee_variables_v2 import obtener_variables, alerta_proactiva

                datos_gee  = obtener_variables(lat, lon, dias_atras=30)
                satelital  = datos_gee
                alerta_gee = alerta_proactiva(datos_gee)

                # Advertencia cruzada: GEE detecta riesgo aunque la foto este sana
                nivel_gee  = datos_gee.get("nivel_riesgo_fusarium", "BAJO")
                clase_foto = res_modelos["clase"]
                if nivel_gee in ["ALTO", "CRITICO"] and clase_foto in ["Sanas", "Sano"]:
                    advertencia = (
                        f"Las condiciones ambientales del lote son de riesgo {nivel_gee} "
                        f"para Fusarium aunque la foto no muestre sintomas visibles. "
                        f"Monitoreo intensivo recomendado en los proximos dias."
                    )

            except ImportError:
                satelital = {"error": "Modulo gee_variables_v2.py no encontrado"}
            except Exception as e:
                satelital = {"error": f"Error consultando GEE: {str(e)}"}
        else:
            satelital = {"nota": "Manda lat y lon para obtener datos satelitales del lote"}

        return {
            **res_modelos,
            "satelital":           satelital,
            "alerta_gee":          alerta_gee,
            "advertencia_cruzada": advertencia,
            "tiempo_ms":           round((time.time() - t0) * 1000, 1),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
