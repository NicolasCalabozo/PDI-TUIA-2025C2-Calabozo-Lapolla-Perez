import numpy as np
import matplotlib.pyplot as plt
import cv2

def encontrar_tramos(bin_linea: np.ndarray, min_len: int = 10):
    """
    bin_linea: vector booleano 1D (fila o columna de la máscara)
    min_len: longitud mínima del tramo para considerarlo línea real
    Devuelve: lista de (inicio, fin) inclusive-exclusivo en índices
    """
    # Convertimos a int para usar diff
    a = bin_linea.astype(np.uint8)
    # Detectar bordes de tramos: 0->1 (inicio), 1->0 (fin)
    da = np.diff(a, prepend=0, append=0)
    inicios = np.where(da == 1)[0]
    fines   = np.where(da == -1)[0]
    # Filtrar por longitud mínima
    tramos = []
    for s, e in zip(inicios, fines):
        if (e - s) >= min_len:
            tramos.append((s, e))  # [s, e)
    return tramos

if __name__ == '__main__':
    img = cv2.imread(filename='formulario_vacio.png', flags=cv2.IMREAD_GRAYSCALE)
    img_para_dibujar = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 1) Umbral simple (líneas oscuras sobre fondo claro)
    mascara_umbral = img < 160  # bool

    # 2) Proyecciones para elegir filas/columnas candidatas
    proyeccion_vertical = np.sum(mascara_umbral, axis=0)   # por columnas (x)
    proyeccion_horizontal = np.sum(mascara_umbral, axis=1) # por filas (y)

    umbral_vertical = 170
    umbral_horizontal = 250

    coord_x_lineas = np.where(proyeccion_vertical > umbral_vertical)[0]  # columnas candidatas
    coord_y_lineas = np.where(proyeccion_horizontal > umbral_horizontal)[0]  # filas candidatas

    alto, ancho, _ = img_para_dibujar.shape
    color_rojo = (0, 0, 255)
    grosor_linea = 1

    # Parámetros de limpieza para evitar ruiditos
    # Longitud mínima de segmento (en píxeles)
    min_len_h = 20   # horizontales
    min_len_v = 20   # verticales

    # 3) Buscar tramos por FILA (horizontales)
    segmentos_h = []  # (x1, x2, y)
    for y in coord_y_lineas:
        fila = mascara_umbral[y, :]              # bool 1D (ancho)
        tramos = encontrar_tramos(fila, min_len=min_len_h)
        for x1, x2 in tramos:
            # Guardar y dibujar sólo ese tramo
            segmentos_h.append((x1, x2, y))
            cv2.line(img_para_dibujar, (int(x1), int(y)), (int(x2-1), int(y)), color_rojo, grosor_linea)

    # 4) Buscar tramos por COLUMNA (verticales)
    segmentos_v = []  # (x, y1, y2)
    for x in coord_x_lineas:
        col = mascara_umbral[:, x]               # bool 1D (alto)
        tramos = encontrar_tramos(col, min_len=min_len_v)
        for y1, y2 in tramos:
            segmentos_v.append((x, y1, y2))
            cv2.line(img_para_dibujar, (int(x), int(y1)), (int(x), int(y2-1)), color_rojo, grosor_linea)

    # 5) (Opcional) Si querés medir longitudes:
    longitudes_h = [(x2 - x1) for (x1, x2, _) in segmentos_h]
    longitudes_v = [(y2 - y1) for (_, y1, y2) in segmentos_v]

    # 6) Mostrar resultado
    plt.figure(figsize=(12, 12))
    plt.imshow(cv2.cvtColor(img_para_dibujar, cv2.COLOR_BGR2RGB))
    plt.title('Segmentos detectados (tramos contiguos) en vez de líneas completas')
    plt.axis('off')
    plt.show()

    # 7) (Opcional) Tenés posiciones y longitudes en estas listas:
    # - segmentos_h: lista de (x_inicio, x_fin_exclusivo, y_constante)
    # - segmentos_v: lista de (x_constante, y_inicio, y_fin_exclusivo)
    # - longitudes_h / longitudes_v: métricas por segmento
