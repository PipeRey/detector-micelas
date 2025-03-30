"""
Micela Detector App (Banner Profesional)
-----------------------------------------
App con banner superior, fuente centrada y visualización mejorada.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64

st.set_page_config(page_title='Micela Detector', layout='centered')
st.title('🔬 Detector de Micelas')
st.write('Sube una imagen en escala de grises para detectar micelas.')

uploaded_file = st.file_uploader("Selecciona una imagen", type=["png", "jpg", "jpeg"])

def detectar_micelas_ultrasensible(imagen_gris):
    """Detección avanzada de micelas con dibujo de círculos verdes."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrastada = clahe.apply(imagen_gris)
    suavizada = cv2.GaussianBlur(contrastada, (3, 3), 0.5)
    binaria = cv2.adaptiveThreshold(suavizada, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 9, 1)
    kernel = np.ones((1, 1), np.uint8)
    apertura = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=1)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 0.2
    params.maxArea = 1500
    params.minThreshold = 1
    params.maxThreshold = 200
    params.thresholdStep = 2
    params.filterByCircularity = False
    params.filterByInertia = False
    params.filterByConvexity = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(apertura)

    resultado = cv2.cvtColor(imagen_gris, cv2.COLOR_GRAY2BGR)
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        r = int(kp.size / 2)
        cv2.circle(resultado, (x, y), r, (0, 255, 0), 1)

    return resultado, len(keypoints)

def agregar_banner_superior(imagen: np.ndarray, texto: str) -> np.ndarray:
    """Agrega un banner blanco centrado con texto profesional en negro."""
    alto, ancho = imagen.shape[:2]
    banner_alto = 60
    banner = np.ones((banner_alto, ancho, 3), dtype=np.uint8) * 255  # fondo blanco

    # Usar una fuente más profesional
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 1
    thickness = 2
    text_size = cv2.getTextSize(texto, font, font_scale, thickness)[0]
    text_x = (ancho - text_size[0]) // 2
    text_y = (banner_alto + text_size[1]) // 2

    cv2.putText(banner, texto, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
    return np.vstack((banner, imagen))

def convertir_a_descarga(imagen):
    """Convierte imagen BGR a enlace HTML para descarga como PNG."""
    _, buffer = cv2.imencode('.png', imagen)
    b64 = base64.b64encode(buffer).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="micelas_resultado.png">📥 Descargar imagen procesada</a>'
    return href

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    img_array = np.array(image)

    # Mostrar encabezado encima de la imagen original
    st.markdown("#### Imagen original")
    st.image(image, use_container_width=True)

    with st.spinner('Detectando micelas...'):
        resultado_img, total = detectar_micelas_ultrasensible(img_array)
        resultado_final = agregar_banner_superior(resultado_img, f"Micelas detectadas: {total}")
        st.markdown("#### Imagen procesada")
        st.image(resultado_final, use_container_width=True)
        st.markdown(convertir_a_descarga(resultado_final), unsafe_allow_html=True)
        st.success(f"✔ Se detectaron {total} micelas.")


