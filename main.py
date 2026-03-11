"""
BananaVision API — main.py
==========================
Correr: uvicorn main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /                      → health check
    POST /enfermedades          → Modelo A solo
    POST /nutrientes            → Modelo B solo
    POST /diagnostico           → Pipeline completo (A → si sana → B)
    POST /diagnostico/completo  → A + B + Google Earth Engine
"""

import io
import os
import time
import urllib.request

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

URL_ENF  = "https://storage.googleapis.com/modelos_sala_ai/modelo_enfermedades_FINAL.keras"
URL_NUT  = "https://storage.googleapis.com/modelos_sala_ai/nutricion_v2_FINAL.keras"

IMG_SIZE = (224, 224)
UMBRAL   = 0.45

CLASES_ENF = ["Cordana", "Fusarium", "Sanas", "SigatokaNegra"]
CLASES_NUT = ["Boron", "Calcium", "Iron", "Magnesium",
              "Manganese", "Potassium", "Sano", "Sulphur"]

INFO_ENF = {
    "Cordana":       {"urgencia": "MEDIA",      "accion": "Deshoje sanitario. Aplicar fungicida cuprico preventivo.",          "exportable": "SI (con monitoreo activo)",          "recuperable": "SI"},
    "Fusarium":      {"urgencia": "CRITICA",     "accion": "CUARENTENA TOTAL. Aislar lote. Llamar a AGROCALIDAD: 1800-AGRO.", "exportable": "NO — suelo inutilizable 30-40 anos",  "recuperable": "NO"},
    "Sanas":         {"urgencia": "BAJA",        "accion": "Planta sana. Continuar monitoreo semanal.",                        "exportable": "SI",                                 "recuperable": "N/A"},
    "SigatokaNegra": {"urgencia": "MEDIA-ALTA",  "accion": "Deshoje urgente. Fungicida sistemico antes de la proxima lluvia.","exportable": "SI (riesgo maduracion prematura)",   "recuperable": "SI (requiere control permanente)"},
}

INFO_NUT = {
    "Boron":     {"sintoma": "Hojas jovenes deformadas, puntas necroticas",     "tratamiento": "Borax foliar 1-2 g/L. Aplicar en horas frescas.",        "producto_tipo": "Fertilizante foliar — Boro"},
    "Calcium":   {"sintoma": "Bordes foliares secos, necrosis marginal",         "tratamiento": "Nitrato de calcio foliar 0.5%",                           "producto_tipo": "Fertilizante foliar — Calcio"},
    "Iron":      {"sintoma": "Clorosis internervial en hojas jovenes",           "tratamiento": "Quelato de hierro foliar (verificar pH < 6.5)",            "producto_tipo": "Quelato de hierro"},
    "Magnesium": {"sintoma": "Clorosis entre nervaduras en hojas viejas",        "tratamiento": "Sulfato de magnesio MgSO4 foliar 1%",                     "producto_tipo": "Fertilizante — Magnesio"},
    "Manganese": {"sintoma": "Manchas cloroticas irregulares, aspecto moteado", "tratamiento": "Sulfato de manganeso foliar en horas frescas",            "producto_tipo": "Fertilizante foliar — Manganeso"},
    "Potassium": {"sintoma": "Bordes quemados, fruta pequena, hojas viejas",    "tratamiento": "KCl o K2SO4 al suelo segun analisis",                    "producto_tipo": "Fertilizante suelo — Potasio"},
    "Sano":      {"sintoma": "Planta en estado nutricional optimo",              "tratamiento": "Mantener plan de fertilizacion actual",                   "producto_tipo": "N/A"},
    "Sulphur":   {"sintoma": "Clorosis general en hojas jovenes",                "tratamiento": "Sulfato de amonio o azufre elemental al suelo",           "producto_tipo": "Fertilizante — Azufre"},
}


# ──────────────────────────────────────────────────────────────────────────────
# DESCARGA DE MODELOS DESDE GCS PUBLICO (sin credenciales)
# ──────────────────────────────────────────────────────────────────────────────
def _descargar_modelo(ruta_local: str, url: str):
    """
    Descarga el modelo desde una URL pública de GCS.
    Si el archivo ya existe y pesa >= 1 MB se considera válido y se omite.
    Si pesa < 1 MB se asume que es un puntero de Git LFS y se sobreescribe.
    """
    os.makedirs(os.path.dirname(ruta_local), exist_ok=True)

    if os.path.exists(ruta_local):
        size_mb = os.path.getsize(ruta_local) / (1024 * 1024)
        if size_mb >= 1.0:
            print(f"  ✓ {ruta_local} ya existe ({size_mb:.1f} MB), omitiendo descarga.")
            return
        print(f"  ! {ruta_local} parece un puntero LFS ({size_mb:.2f} MB), descargando modelo real...")

    print(f"  ↓ Descargando desde GCS...")
    print(f"    {url}")

    def _progreso(count, block_size, total_size):
        if total_size > 0:
            pct = min(count * block_size * 100 / total_size, 100)
            mb  = count * block_size / (1024 * 1024)
            print(f"\r    {mb:.1f} MB  ({pct:.0f}%)", end="", flush=True)

    urllib.request.urlretrieve(url, ruta_local, reporthook=_progreso)
    size_mb = os.path.getsize(ruta_local) / (1024 * 1024)
    print(f"\n  ✓ Listo: {size_mb:.1f} MB guardado en {ruta_local}")


# ──────────────────────────────────────────────────────────────────────────────
# ARRANQUE: descargar modelos y cargarlos en memoria
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("BananaVision API — iniciando...")
print("=" * 60)

t0 = time.time()

print("Verificando modelos...")
_descargar_modelo(RUTA_ENF, URL_ENF)
_descargar_modelo(RUTA_NUT, URL_NUT)

print("Cargando modelos en TensorFlow...")
modelo_enf = tf.keras.models.load_model(RUTA_ENF)
print(f"  ✓ Modelo A (Enfermedades) listo — {time.time() - t0:.1f}s")

t1 = time.time()
modelo_nut = tf.keras.models.load_model(RUTA_NUT)
print(f"  ✓ Modelo B (Nutrientes)   listo — {time.time() - t1:.1f}s")
print(f"Ambos modelos en memoria ({time.time() - t0:.1f}s total)")
print("=" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIONES INTERNAS
# ──────────────────────────────────────────────────────────────────────────────
def bytes_a_tensor(imagen_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(imagen_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def predecir(modelo, tensor: np.ndarray, clases: list) -> dict:
    preds     = modelo.predict(tensor, verbose=0)[0]
    idx       = int(np.argmax(preds))
    confianza = float(preds[idx])
    return {
        "clase":            clases[idx],
        "confianza":        round(confianza * 100, 1),
        "confiable":        confianza >= UMBRAL,
        "todas_las_clases": {c: round(float(p) * 100, 2) for c, p in zip(clases, preds)},
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
        },
    }


@app.post("/enfermedades")
async def detectar_enfermedad(foto: UploadFile = File(...)):
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


@app.post("/nutrientes")
async def detectar_nutriente(foto: UploadFile = File(...)):
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


@app.post("/diagnostico")
async def diagnostico_completo(foto: UploadFile = File(...)):
    validar_imagen(foto)
    try:
        t0           = time.time()
        imagen_bytes = await foto.read()
        tensor       = bytes_a_tensor(imagen_bytes)

        res_enf         = predecir(modelo_enf, tensor, CLASES_ENF)
        res_enf["info"] = INFO_ENF.get(res_enf["clase"], {})

        if res_enf["clase"] != "Sanas":
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


@app.post("/diagnostico/completo")
async def diagnostico_con_gee(
    foto: UploadFile = File(...),
    lat: float = None,
    lon: float = None,
):
    validar_imagen(foto)
    try:
        t0           = time.time()
        imagen_bytes = await foto.read()
        tensor       = bytes_a_tensor(imagen_bytes)

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

        satelital   = None
        alerta_gee  = None
        advertencia = None

        if lat is not None and lon is not None:
            try:
                from gee_variables_v2 import obtener_variables, alerta_proactiva
                datos_gee  = obtener_variables(lat, lon, dias_atras=30)
                satelital  = datos_gee
                alerta_gee = alerta_proactiva(datos_gee)
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