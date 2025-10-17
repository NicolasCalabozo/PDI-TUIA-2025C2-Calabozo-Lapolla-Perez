import numpy as np
import matplotlib.pyplot as plt
import cv2

def extraer_componentes(celda, th_area_frac=0.002):
    """
    Etiqueta y filtra componentes pequeñas.
    th_area_frac es fracción del área de la celda para definir umbral mínimo.
    Devuelve (stats_filtrados_ordenados_x, centroids_filtrados).
    """
    h, w = celda.shape
    num, labels, stats, cents = cv2.connectedComponentsWithStats(celda, 8, cv2.CV_32S) #type: ignore

    # stats: [x, y, width, height, area]; fila 0 = fondo
    stats = stats[1:]
    cents = cents[1:]

    th_area = th_area_frac * (h * w)
    ix = stats[:, -1] > th_area

    stats_f = stats[ix]
    cents_f = cents[ix]

    # Ordenar por x (izq->der)
    order = np.argsort(stats_f[:, 0])
    return stats_f[order], cents_f[order]

def contar_palabras_y_chars(stats, gap_factor=1.8):
    """
    A partir de las bounding boxes (ordenadas por x), separa en 'palabras'
    usando un gap grande respecto del gap mediano.
    Devuelve (n_palabras, n_chars, cortes_palabras)
    """
    if len(stats) == 0:
        return 0, 0, []

    # extremos x de cada componente
    xs = stats[:, 0]
    xe = stats[:, 0] + stats[:, 2]

    # gaps entre el fin de una y el inicio de la siguiente
    gaps = xs[1:] - xe[:-1]
    if len(gaps) == 0:
        return 1, len(stats), [(0, len(stats))]

    med = np.median(gaps)
    umbral_gap = gap_factor * med if med > 0 else np.max(gaps) + 1

    cortes = [0]
    for i, g in enumerate(gaps, start=1):
        if g > umbral_gap:
            cortes.append(i)
    cortes.append(len(stats))

    tramos = list(zip(cortes[:-1], cortes[1:]))
    n_palabras = len(tramos)
    n_chars = len(stats)
    return n_palabras, n_chars, tramos

def binarizar_celda(celda_img):
    """Devuelve binaria con tinta=1 (blanco) y fondo=0 (negro)."""
    # Otsu sobre invertida para que la “tinta” quede alta
    # (si tu fondo es claro y la tinta oscura, invertimos así).
    _, th = cv2.threshold(255 - celda_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (th > 0).astype(np.uint8)

# b = extraer_componentes(binarizar_celda(celdas['pregunta1_si']))
# palabras = contar_palabras_y_chars(b[0])
# print('----------------------------------')
# print(f'Formulario: {formulario}')
# print(f"Número de palabras en 'nombre_valor': {palabras[0]}")
# print(f"Número de caracteres en 'nombre_valor': {palabras[1]}")
# print(f"Cortes de palabras (índices de caracteres): {palabras}")