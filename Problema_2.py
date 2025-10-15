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


def encontrar_segmentos(mascara_umbral, coordenadas, eje='horizontal', min_largo=10):
    """
    Encuentra segmentos de línea a lo largo de un eje específico,
    filtrando aquellos que son más cortos que min_largo.
    """
    segmentos_dict = {}
    
    for coord in coordenadas:
        if eje == 'horizontal':
            linea_actual = mascara_umbral[coord, :]
        elif eje == 'vertical':
            linea_actual = mascara_umbral[:, coord]
        segmentos_en_linea = []
        en_segmento = False
        inicio = 0

        for i, pixel_es_negro in enumerate(linea_actual):
            if pixel_es_negro and not en_segmento:
                en_segmento = True
                inicio = i
            elif not pixel_es_negro and en_segmento:
                en_segmento = False
                final = i - 1
                if (final - inicio) >= min_largo:
                    segmentos_en_linea.append((inicio, final))
        if en_segmento:
            final = len(linea_actual) - 1
            if (final - inicio) >= min_largo:
                segmentos_en_linea.append((inicio, final))
        
        # Opcional: solo añadimos la coordenada al diccionario si encontramos segmentos válidos
        if segmentos_en_linea:
            segmentos_dict[coord] = segmentos_en_linea
            
    return segmentos_dict

def encontrar_intersecciones(segmentos_hor, segmentos_ver):
    """
    Encuentra los puntos (x, y) donde los segmentos horizontales y verticales se cruzan.
    """
    vertices = []
    # Iteramos sobre cada línea horizontal y sus segmentos
    for y, segs_horizontales in segmentos_hor.items():
        for x_start, x_end in segs_horizontales:
            # Ahora, iteramos sobre cada línea vertical y sus segmentos
            for x, segs_verticales in segmentos_ver.items():
                for y_start, y_end in segs_verticales:
                    
                    # Esta es la condición clave:
                    if (x_start <= x <= x_end) and (y_start <= y <= y_end):
                        vertices.append((x, y))
                        
    return vertices

if __name__ == '__main__':
    figura = True
    segmentos = False
    vertices = True
    img, mascara, vert, hor = detectar_lineas('formulario_01.png', 180, 170, 200)
    img_dummy = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    segmentos_ver = encontrar_segmentos(mascara,vert, 'vertical')
    segmentos_hor = encontrar_segmentos(mascara,hor, 'horizontal')
    vertices = encontrar_intersecciones(segmentos_hor, segmentos_ver)
    color_horizontal = (255, 0, 0)  # Azul
    grosor_linea = 1
    
    if(segmentos):
        for y, segs in segmentos_hor.items():
            for x_start, x_end in segs:
                cv2.line(img_dummy, (x_start, y), (x_end, y), color_horizontal, grosor_linea)

        color_vertical = (0, 0, 255)  # Rojo
        for x, segs in segmentos_ver.items():
            for y_start, y_end in segs:
                cv2.line(img_dummy, (x, y_start), (x, y_end), color_vertical, grosor_linea)
    if(vertices):    
        for x, y in vertices:
            # Dibujamos un pequeño círculo verde en cada vértice
            cv2.circle(img_dummy, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
            
    if(figura):
        plt.figure(figsize=(12, 12))
        # Convertimos de BGR (OpenCV) a RGB (Matplotlib) para mostrar colores correctos
        plt.imshow(cv2.cvtColor(img_dummy, cv2.COLOR_BGR2RGB))
        plt.title('Todos los Segmentos Detectados sobre la Imagen Original')
        plt.axis('off')
        plt.show()