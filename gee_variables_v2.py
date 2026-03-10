"""
gee_variables_v2.py
===================
Modulo que extrae variables ambientales de Google Earth Engine para un punto GPS.
Version 2: compatible con el pipeline combinado BananaVision (Modelo A + Modelo B).

Cambios vs v1:
- Indice de riesgo Fusarium mejorado con peso diferenciado por variable
- Funcion resumen_para_llm() que formatea los datos para prompt de LLM
- Funcion alerta_proactiva() que retorna alerta en lenguaje natural
- Fallbacks mas robustos en todas las capas
- Soporte para multiples puntos GPS (modo lote)

Uso:
    from gee_variables_v2 import obtener_variables, resumen_para_llm
    datos = obtener_variables(lat=-2.09, lon=-79.54, dias_atras=30)
    texto = resumen_para_llm(datos)
"""

import ee
import datetime


# ee.Authenticate()  # Solo la primera vez en una maquina nueva
ee.Initialize(project='salaia')  # Reemplaza con tu proyecto GCP si cambia


# ─────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────
CLASES_SUELO = {
    1: 'Cl - Arcilloso',
    2: 'SiCl - Arcillo-limoso',
    3: 'SaCl - Arcillo-arenoso',
    4: 'ClLo - Franco-arcilloso',
    5: 'SiClLo - Franco-arcillo-limoso',
    6: 'SaClLo - Franco-arcillo-arenoso',
    7: 'Lo - Franco',
    8: 'SiLo - Franco-limoso',
    9: 'SaLo - Franco-arenoso',
    10: 'Si - Limoso',
    11: 'LoSa - Franco-arenoso fino',
    12: 'Sa - Arenoso',
}

# Suelos argilosos tienen mayor riesgo de acumulacion de agua → Fusarium
SUELOS_RIESGO_ALTO = {1, 2, 3, 4, 5}


def _coleccion_vacia(col: ee.ImageCollection) -> bool:
    return col.size().getInfo() == 0


# ─────────────────────────────────────────────────────────────
# FUNCION PRINCIPAL
# ─────────────────────────────────────────────────────────────
def obtener_variables(lat: float, lon: float, dias_atras: int = 30) -> dict:
    """
    Extrae variables ambientales de GEE para un punto GPS.
    Si no hay datos en la ventana principal, amplia automaticamente hasta 90 dias.

    Args:
        lat:        Latitud decimal (ej: -2.09 para Guayas, Ecuador)
        lon:        Longitud decimal (ej: -79.54 para Guayas, Ecuador)
        dias_atras: Ventana de tiempo hacia atras desde hoy (default: 30 dias)

    Returns:
        dict con todas las variables ambientales + indice de riesgo integrado
    """
    punto  = ee.Geometry.Point([lon, lat])
    buffer = punto.buffer(500)  # 500m: evita que un pixel sin datos bloquee todo

    hoy        = datetime.date.today()
    inicio     = (hoy - datetime.timedelta(days=dias_atras)).isoformat()
    fin        = hoy.isoformat()
    inicio_ext = (hoy - datetime.timedelta(days=90)).isoformat()

    resultados = {
        'lat': lat, 'lon': lon,
        'fecha_inicio': inicio, 'fecha_fin': fin,
        'dias_consultados': dias_atras,
    }

    # ─────────────────────────────────────────
    # 1. PRECIPITACION — GPM IMERG V07
    # ─────────────────────────────────────────
    try:
        def _lluvia(f_ini, f_fin):
            col = (
                ee.ImageCollection("NASA/GPM_L3/IMERG_V07")
                .filterBounds(buffer)
                .filterDate(f_ini, f_fin)
                .select('precipitation')
            )
            total = col.map(lambda img: img.multiply(0.5)).sum()
            return total, col.size().getInfo()

        img_lluvia, n = _lluvia(inicio, fin)
        if n == 0:
            img_lluvia, n = _lluvia(inicio_ext, fin)
            resultados['precipitacion_nota'] = 'Datos extendidos a 90 dias'

        val = img_lluvia.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=buffer, scale=11132
        ).get('precipitation')
        resultados['precipitacion_mm'] = round(ee.Number(val).getInfo(), 2)
    except Exception as e:
        resultados['precipitacion_mm'] = None
        resultados['error_precipitacion'] = str(e)

    # ─────────────────────────────────────────
    # 2. HUMEDAD DEL SUELO — SMAP con fallback ERA5
    # ─────────────────────────────────────────
    try:
        smap = (
            ee.ImageCollection("NASA/SMAP/SPL4SMGP/008")
            .filterBounds(buffer).filterDate(inicio, fin).select('sm_surface')
        )
        if _coleccion_vacia(smap):
            smap = (
                ee.ImageCollection("NASA/SMAP/SPL4SMGP/008")
                .filterBounds(buffer).filterDate(inicio_ext, fin).select('sm_surface')
            )
        val_smap = smap.mean().reduceRegion(
            reducer=ee.Reducer.mean(), geometry=buffer, scale=11000
        ).get('sm_surface')
        resultados['humedad_suelo_m3m3'] = round(ee.Number(val_smap).getInfo(), 4)
        resultados['humedad_fuente'] = 'SMAP'
    except Exception:
        try:
            era5 = (
                ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
                .filterBounds(buffer).filterDate(inicio, fin)
                .select('volumetric_soil_water_layer_1').mean()
            )
            val = era5.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=buffer, scale=11000
            ).get('volumetric_soil_water_layer_1')
            resultados['humedad_suelo_m3m3'] = round(ee.Number(val).getInfo(), 4)
            resultados['humedad_fuente'] = 'ERA5-Land (fallback)'
        except Exception as e2:
            resultados['humedad_suelo_m3m3'] = None
            resultados['error_humedad'] = str(e2)

    # ─────────────────────────────────────────
    # 3. TEMPERATURA — MODIS LST con fallback ERA5
    # ─────────────────────────────────────────
    try:
        modis = (
            ee.ImageCollection("MODIS/061/MOD11A1")
            .filterBounds(buffer).filterDate(inicio, fin).select('LST_Day_1km')
        )
        if _coleccion_vacia(modis):
            modis = (
                ee.ImageCollection("MODIS/061/MOD11A1")
                .filterBounds(buffer).filterDate(inicio_ext, fin).select('LST_Day_1km')
            )
        temp_raw = modis.mean().reduceRegion(
            reducer=ee.Reducer.mean(), geometry=buffer, scale=1000
        ).get('LST_Day_1km')
        temp_c = ee.Number(temp_raw).multiply(0.02).subtract(273.15)
        resultados['temperatura_c'] = round(temp_c.getInfo(), 2)
        resultados['temperatura_fuente'] = 'MODIS LST'
    except Exception:
        try:
            era5_t = (
                ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
                .filterBounds(buffer).filterDate(inicio, fin)
                .select('temperature_2m').mean()
            )
            t_raw = era5_t.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=buffer, scale=11000
            ).get('temperature_2m')
            t_c = ee.Number(t_raw).subtract(273.15)
            resultados['temperatura_c'] = round(t_c.getInfo(), 2)
            resultados['temperatura_fuente'] = 'ERA5-Land (fallback)'
        except Exception as e2:
            resultados['temperatura_c'] = None
            resultados['error_temperatura'] = str(e2)

    # ─────────────────────────────────────────
    # 4. NDVI + NDRE — Sentinel-2 con fallback MODIS
    # ─────────────────────────────────────────
    try:
        def _s2(f_ini, f_fin, nubes=60):
            return (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(buffer).filterDate(f_ini, f_fin)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', nubes))
            )

        s2_col = _s2(inicio, fin)
        n_s2 = s2_col.size().getInfo()
        if n_s2 == 0:
            s2_col = _s2(inicio_ext, fin, nubes=80)
            n_s2 = s2_col.size().getInfo()
            resultados['s2_nota'] = f'Extendido a 90 dias, {n_s2} imagenes'

        if n_s2 > 0:
            s2   = s2_col.median()
            ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')
            ndre = s2.normalizedDifference(['B8', 'B5']).rename('NDRE')
            vals = ndvi.addBands(ndre).reduceRegion(
                reducer=ee.Reducer.mean(), geometry=buffer, scale=20
            )
            resultados['ndvi'] = round(ee.Number(vals.get('NDVI')).getInfo(), 4)
            resultados['ndre'] = round(ee.Number(vals.get('NDRE')).getInfo(), 4)
            resultados['ndvi_fuente'] = 'Sentinel-2'
        else:
            modis_ndvi = (
                ee.ImageCollection("MODIS/061/MOD13Q1")
                .filterBounds(buffer).filterDate(inicio_ext, fin).select('NDVI').mean()
            )
            val_ndvi = modis_ndvi.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=buffer, scale=250
            ).get('NDVI')
            resultados['ndvi'] = round(ee.Number(val_ndvi).multiply(0.0001).getInfo(), 4)
            resultados['ndre'] = None
            resultados['ndvi_fuente'] = 'MODIS MOD13Q1 (fallback, sin NDRE)'

        # Alerta de vegetacion basada en NDRE o NDVI
        ref = resultados.get('ndre') or resultados.get('ndvi')
        if ref is not None:
            if ref < 0.20:
                resultados['alerta_vegetacion'] = 'CRITICO - Estres severo o enfermedad activa'
            elif ref < 0.35:
                resultados['alerta_vegetacion'] = 'ATENCION - Estres moderado detectado'
            else:
                resultados['alerta_vegetacion'] = 'NORMAL'

    except Exception as e:
        resultados['ndvi'] = None; resultados['ndre'] = None
        resultados['error_sentinel2'] = str(e)

    # ─────────────────────────────────────────
    # 5. RADAR INUNDACION — Sentinel-1 SAR
    # ─────────────────────────────────────────
    try:
        def _s1(f_ini, f_fin):
            return (
                ee.ImageCollection('COPERNICUS/S1_GRD')
                .filterBounds(buffer).filterDate(f_ini, f_fin)
                .filter(ee.Filter.eq('instrumentMode', 'IW'))
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                .select('VV')
            )

        s1 = _s1(inicio, fin)
        if _coleccion_vacia(s1):
            s1 = _s1(inicio_ext, fin)

        val = s1.mean().reduceRegion(
            reducer=ee.Reducer.mean(), geometry=buffer, scale=30
        ).get('VV')
        db = ee.Number(val).getInfo()
        resultados['radar_vv_db'] = round(db, 2) if db is not None else None
        if db is not None:
            resultados['riesgo_inundacion'] = (
                'ALTO'  if db < -15 else
                'MEDIO' if db < -12 else
                'BAJO'
            )
    except Exception as e:
        resultados['radar_vv_db'] = None
        resultados['error_radar'] = str(e)

    # ─────────────────────────────────────────
    # 6. ELEVACION Y PENDIENTE — SRTM 30m
    # ─────────────────────────────────────────
    try:
        srtm    = ee.Image("USGS/SRTMGL1_003")
        terrain = ee.Terrain.products(srtm)
        vals = terrain.select(['elevation', 'slope', 'aspect']).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=buffer, scale=30
        )
        resultados['elevacion_m']   = round(ee.Number(vals.get('elevation')).getInfo(), 1)
        resultados['pendiente_deg'] = round(ee.Number(vals.get('slope')).getInfo(), 2)
        resultados['aspecto_deg']   = round(ee.Number(vals.get('aspect')).getInfo(), 1)
    except Exception as e:
        resultados['elevacion_m'] = None
        resultados['error_terreno'] = str(e)

    # ─────────────────────────────────────────
    # 7. TIPO DE SUELO — OpenLandMap
    # ─────────────────────────────────────────
    try:
        suelo_img = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select('b0')
        val_suelo = suelo_img.reduceRegion(
            reducer=ee.Reducer.mode(), geometry=buffer, scale=250
        ).get('b0')
        codigo = int(ee.Number(val_suelo).getInfo() or 0)
        resultados['tipo_suelo_codigo'] = codigo
        resultados['tipo_suelo_nombre'] = CLASES_SUELO.get(codigo, 'Desconocido')
        resultados['riesgo_suelo_fusarium'] = 'ALTO' if codigo in SUELOS_RIESGO_ALTO else 'BAJO'
    except Exception as e:
        resultados['tipo_suelo_codigo'] = None
        resultados['error_suelo'] = str(e)

    # ─────────────────────────────────────────
    # INDICE DE RIESGO AMBIENTAL INTEGRADO v2
    # Score 0-12 con pesos diferenciados por variable
    # ─────────────────────────────────────────
    try:
        score = 0

        # Precipitacion (max 3 puntos)
        lluvia = resultados.get('precipitacion_mm') or 0
        if lluvia > 200:   score += 3
        elif lluvia > 100: score += 2
        elif lluvia > 50:  score += 1

        # Radar inundacion (max 3 puntos — peso alto: agua en suelo = Fusarium)
        radar_r = resultados.get('riesgo_inundacion', 'BAJO')
        if radar_r == 'ALTO':   score += 3
        elif radar_r == 'MEDIO': score += 1

        # Tipo de suelo (max 2 puntos — argiloso retiene agua)
        suelo_r = resultados.get('riesgo_suelo_fusarium', 'BAJO')
        if suelo_r == 'ALTO': score += 2

        # NDVI bajo (max 2 puntos — planta estresada = mas susceptible)
        ndvi_v = resultados.get('ndvi') or 1.0
        if ndvi_v < 0.20:   score += 2
        elif ndvi_v < 0.35: score += 1

        # Temperatura (max 2 puntos — 25-30C es optimo para Fusarium)
        temp = resultados.get('temperatura_c') or 0
        if 22 <= temp <= 32: score += 2
        elif 18 <= temp < 22: score += 1

        resultados['indice_riesgo_ambiental'] = score
        resultados['nivel_riesgo_fusarium'] = (
            'CRITICO' if score >= 9 else
            'ALTO'    if score >= 6 else
            'MEDIO'   if score >= 3 else
            'BAJO'
        )
    except Exception:
        pass

    return resultados


# ─────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES PARA EL PIPELINE
# ─────────────────────────────────────────────────────────────

def resumen_para_llm(datos: dict) -> str:
    """
    Formatea los datos GEE en un parrafo estructurado listo para
    incluirlo en el prompt de un LLM (Claude API, GPT-4o, etc.).

    Returns:
        str con resumen en espanol de las condiciones ambientales del lote
    """
    lineas = [
        f"Condiciones ambientales del lote (lat={datos.get('lat')}, lon={datos.get('lon')}):",
        f"- Periodo consultado: {datos.get('fecha_inicio')} al {datos.get('fecha_fin')}",
        f"- Precipitacion acumulada: {datos.get('precipitacion_mm', 'No disponible')} mm",
        f"- Humedad del suelo: {datos.get('humedad_suelo_m3m3', 'No disponible')} m3/m3 (fuente: {datos.get('humedad_fuente','N/A')})",
        f"- Temperatura promedio: {datos.get('temperatura_c', 'No disponible')} grados C",
        f"- NDVI (salud vegetal): {datos.get('ndvi', 'No disponible')} | NDRE: {datos.get('ndre', 'No disponible')}",
        f"- Alerta vegetacion: {datos.get('alerta_vegetacion', 'No disponible')}",
        f"- Riesgo de inundacion (SAR): {datos.get('riesgo_inundacion', 'No disponible')}",
        f"- Elevacion: {datos.get('elevacion_m', 'No disponible')} m | Pendiente: {datos.get('pendiente_deg', 'N/A')} grados",
        f"- Tipo de suelo: {datos.get('tipo_suelo_nombre', 'No disponible')}",
        f"- Riesgo suelo para Fusarium: {datos.get('riesgo_suelo_fusarium', 'No disponible')}",
        f"- Indice de riesgo ambiental integrado: {datos.get('indice_riesgo_ambiental', 'N/A')}/12",
        f"- Nivel de riesgo Fusarium ambiental: {datos.get('nivel_riesgo_fusarium', 'No disponible')}",
    ]
    return '\n'.join(lineas)


def alerta_proactiva(datos: dict) -> dict:
    """
    Genera una alerta proactiva en lenguaje natural basada en el
    indice de riesgo ambiental, sin necesidad de foto del agricultor.
    Util para notificaciones push en la app movil.

    Returns:
        dict con nivel de alerta, titulo y mensaje para notificacion
    """
    nivel = datos.get('nivel_riesgo_fusarium', 'BAJO')
    score = datos.get('indice_riesgo_ambiental', 0)
    lluvia = datos.get('precipitacion_mm', 0) or 0
    radar  = datos.get('riesgo_inundacion', 'BAJO')

    if nivel == 'CRITICO':
        return {
            'nivel': 'CRITICO',
            'titulo': 'Alerta maxima: condiciones criticas para Fusarium',
            'mensaje': (
                f"Las condiciones ambientales de tu lote son de muy alto riesgo para Fusarium "
                f"(score {score}/12). Precipitacion de {lluvia:.0f}mm, suelo saturado y temperatura "
                f"optima para el hongo. Inspecciona tus plantas hoy y evita mover suelo entre lotes."
            ),
            'accion': 'Inspeccionar lote inmediatamente. Notificar a AGROCALIDAD si detectas sintomas.',
        }
    elif nivel == 'ALTO':
        return {
            'nivel': 'ALTO',
            'titulo': 'Alerta: condiciones favorables para enfermedades fungicas',
            'mensaje': (
                f"Tu lote tiene condiciones de riesgo elevado (score {score}/12). "
                f"La humedad y temperatura actuales favorecen el desarrollo de hongos. "
                f"Considera aplicar fungicida preventivo en los proximos dias."
            ),
            'accion': 'Aplicar fungicida preventivo. Monitorear hojas nuevas.',
        }
    elif nivel == 'MEDIO':
        return {
            'nivel': 'MEDIO',
            'titulo': 'Condiciones de riesgo moderado en tu lote',
            'mensaje': (
                f"Las condiciones actuales presentan riesgo moderado (score {score}/12). "
                f"Mantente atento a cambios en el follaje y realiza un escaneo con BananaVision "
                f"si observas manchas o decoloraciones."
            ),
            'accion': 'Monitoreo semanal recomendado.',
        }
    else:
        return {
            'nivel': 'BAJO',
            'titulo': 'Condiciones ambientales normales',
            'mensaje': f'Tu lote presenta condiciones de riesgo bajo (score {score}/12). Continua con el monitoreo rutinario.',
            'accion': 'Continuar con bioseguridad preventiva normal.',
        }


def obtener_multiples_puntos(coordenadas: list, dias_atras: int = 30) -> list:
    """
    Extrae variables GEE para multiples puntos GPS (modo lote con varios sectores).

    Args:
        coordenadas: lista de tuplas [(lat, lon), (lat, lon), ...]
        dias_atras: ventana de tiempo

    Returns:
        lista de dicts con resultados por punto
    """
    resultados = []
    for lat, lon in coordenadas:
        try:
            datos = obtener_variables(lat, lon, dias_atras)
            resultados.append(datos)
        except Exception as e:
            resultados.append({'lat': lat, 'lon': lon, 'error': str(e)})
    return resultados


# ─────────────────────────────────────────────────────────────
# TEST RAPIDO
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    LAT, LON = -2.0897802, -79.5418231
    print(f"Consultando variables para lat={LAT}, lon={LON}...\n")
    data = obtener_variables(LAT, LON, dias_atras=30)

    print("=== VARIABLES AMBIENTALES ===")
    for k, v in data.items():
        print(f"  {k:<40}: {v}")

    print("\n=== RESUMEN PARA LLM ===")
    print(resumen_para_llm(data))

    print("\n=== ALERTA PROACTIVA ===")
    alerta = alerta_proactiva(data)
    for k, v in alerta.items():
        print(f"  {k}: {v}")
