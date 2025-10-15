import numpy as np
import matplotlib.pyplot as plt
import cv2

def detectar_lineas(nombre_archivo, umbral_mascara, umbral_vert, umbral_hor):
    img = cv2.imread(filename=nombre_archivo, flags=cv2.IMREAD_GRAYSCALE)
    mascara_umbral = img < umbral_mascara #type: ignore
    #Sumamos los True en las columnas para generar un vector que nos diga
    #en qué coordenada X hay más concentracion de pixeles "blancos" (con valor 1)
    proyeccion_vertical = np.sum(mascara_umbral, axis=0)
    proyeccion_horizontal = np.sum(mascara_umbral, axis=1)
    coord_x_lineas = list(np.where(proyeccion_vertical > umbral_vert)[0])
    coord_y_lineas = list(np.where(proyeccion_horizontal > umbral_hor)[0])
    x_unicas = coord_x_lineas[1::2]
    y_unicas = coord_y_lineas[1::2]
    #return lineas verticales, lineas horizontales
    return img, mascara_umbral, x_unicas, y_unicas


def encontrar_segmentos_verticales(mascara_umbral, lineas_x):
    """
    Itera sobre cada coordenada X de una línea y encuentra los
    segmentos de línea (y_inicio, y_final) en esa columna.
    """
    segmentos_por_columna = {}
    for x in lineas_x:
        columna_actual = mascara_umbral[:, x]
        segmentos_en_columna = []
        en_segmento = False
        y_inicio = 0
        for y, pixel_es_negro in enumerate(columna_actual):
            if pixel_es_negro and not en_segmento:
                en_segmento = True
                y_inicio = y
            elif not pixel_es_negro and en_segmento:
                en_segmento = False
                y_final = y - 1
                segmentos_en_columna.append((y_inicio, y_final))
        if en_segmento:
            segmentos_en_columna.append((y_inicio, len(columna_actual) - 1))
            
        segmentos_por_columna[x] = segmentos_en_columna
        
    return segmentos_por_columna

def encontrar_segmentos_horizontales(mascara_umbral, lineas_y):
    """
    Itera sobre cada coordenada Y de una línea y encuentra los
    segmentos de línea (x_inicio, x_final) en esa fila.
    """
    segmentos_por_fila = {}
    for y in lineas_y:
        fila_actual = mascara_umbral[y, :]
        segmentos_en_fila = []
        en_segmento = False
        x_inicio = 0

        for x, pixel_es_negro in enumerate(fila_actual):
            if pixel_es_negro and not en_segmento:
                en_segmento = True
                x_inicio = x
            elif not pixel_es_negro and en_segmento:
                en_segmento = False
                x_final = x - 1
                segmentos_en_fila.append((x_inicio, x_final))
                
        if en_segmento:
            segmentos_en_fila.append((x_inicio, len(fila_actual) - 1))
            
        segmentos_por_fila[y] = segmentos_en_fila
        
    return segmentos_por_fila

if __name__ == '__main__':
    img, mascara, vert, hor = detectar_lineas('formulario_vacio.png', 180, 170, 200)
    img_dummy = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    segmentos_ver = encontrar_segmentos_verticales(mascara,vert)
    segmentos_hor = encontrar_segmentos_horizontales(mascara,hor)
    color_horizontal = (255, 0, 0)  # Azul
    grosor_linea = 2
    for y, segs in segmentos_hor.items():
        for x_start, x_end in segs:
            cv2.line(img_dummy, (x_start, y), (x_end, y), color_horizontal, grosor_linea)

    # 3. Dibujar los segmentos verticales (en rojo)
    color_vertical = (0, 0, 255)  # Rojo
    for x, segs in segmentos_ver.items():
        for y_start, y_end in segs:
            cv2.line(img_dummy, (x, y_start), (x, y_end), color_vertical, grosor_linea)
            
    # 4. Mostrar el resultado final
    plt.figure(figsize=(12, 12))
    # Convertimos de BGR (OpenCV) a RGB (Matplotlib) para mostrar colores correctos
    plt.imshow(cv2.cvtColor(img_dummy, cv2.COLOR_BGR2RGB))
    plt.title('Todos los Segmentos Detectados sobre la Imagen Original')
    plt.axis('off')
    plt.show()