import numpy as np
import matplotlib.pyplot as plt
import cv2


def detectar_lineas(nombre_archivo, umbral_mascara, umbral_vert, umbral_hor):
    img = cv2.imread(filename=nombre_archivo, flags=cv2.IMREAD_GRAYSCALE)
    mascara_umbral = img < umbral_mascara  # type: ignore
    # Sumamos los True en las columnas para generar un vector que nos diga
    # en qué coordenada X hay más concentracion de pixeles "blancos" (con valor 1)
    proyeccion_vertical = np.sum(mascara_umbral, axis=0)
    proyeccion_horizontal = np.sum(mascara_umbral, axis=1)
    coord_x_lineas = list(np.where(proyeccion_vertical > umbral_vert)[0])
    coord_y_lineas = list(np.where(proyeccion_horizontal > umbral_hor)[0])
    x_unicas = coord_x_lineas[1::2]
    y_unicas = coord_y_lineas[1::2]
    # return lineas verticales, lineas horizontales
    return img, mascara_umbral, x_unicas, y_unicas


def encontrar_segmentos(mascara_umbral, coordenadas, eje='horizontal', min_largo=10):
    """
    Encuentra segmentos de línea a lo largo de un eje específico,
    filtrando aquellos que son más cortos que min_largo.
    """
    segmentos_dict = {}
    linea_actual = []
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

def dibujar_segmentos_horizontales(imagen, segmentos_hor, color=(255, 0, 0), grosor=2):
    for y, segs in segmentos_hor.items():
        for x_inicio, x_fin in segs:
            cv2.line(imagen, (x_inicio, y), (x_fin, y), color, grosor)

def dibujar_segmentos_verticales(imagen, segmentos_ver, color=(0, 0, 255), grosor=2):
    for x, segs in segmentos_ver.items():
        for y_inicio, y_fin in segs:
            cv2.line(imagen, (x, y_inicio), (x, y_fin), color, grosor)

def dibujar_vertices(imagen, vertices, color=(0, 255, 0), radio=3):
    for x, y in vertices:
        cv2.circle(imagen, (x, y), radius=radio, color=color, thickness=-1) # thickness=-1 rellena el círculo

def encontrar_celdas(img, segmentos_hor, segmentos_ver, margen=2):
    filas = []
    seg_hor = sorted(list(segmentos_hor.keys()))
    for i in range(len(seg_hor) - 1):
        fila_recortada = img[seg_hor[i] + margen: seg_hor[i+1] - margen, :]
        filas.append(fila_recortada)

    seg_ver = sorted(list(segmentos_ver.keys()))
    celdas = {}

    celdas['titulo'] = filas[0][:, seg_ver[0]+margen: seg_ver[3]-margen]

    celdas['nombre'] = filas[1][:, seg_ver[0]+margen: seg_ver[1]-margen]
    celdas['nombre_valor'] = filas[1][:, seg_ver[1]+margen: seg_ver[3]-margen]

    celdas['edad'] = filas[2][:, seg_ver[0]+margen: seg_ver[1]-margen]
    celdas['edad_valor'] = filas[2][:, seg_ver[1]+margen: seg_ver[3]-margen]

    celdas['mail'] = filas[3][:, seg_ver[0]+margen: seg_ver[1]-margen]
    celdas['mail_valor'] = filas[3][:, seg_ver[1]+margen: seg_ver[3]-margen]

    celdas['legajo'] = filas[4][:, seg_ver[0]+margen: seg_ver[1]-margen]
    celdas['legajo_valor'] = filas[4][:, seg_ver[1]+margen: seg_ver[3]-margen]

    celdas['pregunta1'] = filas[6][:, seg_ver[0]+margen: seg_ver[1]-margen]
    celdas['pregunta1_si'] = filas[6][:, seg_ver[1]+margen: seg_ver[2]-margen]
    celdas['pregunta1_no'] = filas[6][:, seg_ver[2]+margen: seg_ver[3]-margen]

    celdas['pregunta2'] = filas[7][:, seg_ver[0]+margen: seg_ver[1]-margen]
    celdas['pregunta2_si'] = filas[7][:, seg_ver[1]+margen: seg_ver[2]-margen]
    celdas['pregunta2_no'] = filas[7][:, seg_ver[2]+margen: seg_ver[3]-margen]

    celdas['pregunta3'] = filas[8][:, seg_ver[0]+margen: seg_ver[1]-margen]
    celdas['pregunta3_si'] = filas[8][:, seg_ver[1]+margen: seg_ver[2]-margen]
    celdas['pregunta3_no'] = filas[8][:, seg_ver[2]+margen: seg_ver[3]-margen]

    celdas['comentario'] = filas[9][:, seg_ver[0]+margen: seg_ver[1]-margen]
    celdas['comentario_valor'] = filas[9][:,seg_ver[1]+margen: seg_ver[3]-margen]

    return celdas

def mostrar_celda_grilla(ax, titulo, imagen):
    ax.imshow(imagen, cmap='gray', vmin=0, vmax=255)
    ax.set_title(titulo)
    ax.axis('on')
    ax.set_xticks([])
    ax.set_yticks([])

def mostrar_celda(titulo, imagen):
    plt.imshow(imagen, cmap='gray', vmin=0, vmax=255)
    plt.title(titulo)
    plt.show()

    
def mostrar_formulario_desarmado(celdas):
    """
    Muestra todas las celdas extraídas en una grilla que simula el formulario.
    """
    fig, axs = plt.subplots(10, 3, figsize=(10, 15))

    for ax_row in axs:
        for ax in ax_row:
            ax.axis('off')

    mostrar_celda_grilla(axs[1, 0], 'nombre', celdas['nombre'])
    ax_nombre_valor = plt.subplot2grid((10, 3), (1, 1), colspan=2)
    mostrar_celda_grilla(ax_nombre_valor, 'nombre_valor', celdas['nombre_valor'])
    
    mostrar_celda_grilla(axs[2, 0], 'edad', celdas['edad'])
    ax_edad_valor = plt.subplot2grid((10, 3), (2, 1), colspan=2)
    mostrar_celda_grilla(ax_edad_valor, 'edad_valor', celdas['edad_valor'])

    mostrar_celda_grilla(axs[3, 0], 'mail', celdas['mail'])
    ax_mail_valor = plt.subplot2grid((10, 3), (3, 1), colspan=2)
    mostrar_celda_grilla(ax_mail_valor, 'mail_valor', celdas['mail_valor'])
    mostrar_celda_grilla(axs[4, 0], 'legajo', celdas['legajo'])
    ax_legajo_valor = plt.subplot2grid((10, 3), (4, 1), colspan=2)
    mostrar_celda_grilla(ax_legajo_valor, 'legajo_valor', celdas['legajo_valor'])
    mostrar_celda_grilla(axs[6, 0], 'pregunta1', celdas['pregunta1'])
    mostrar_celda_grilla(axs[6, 1], 'pregunta1_si', celdas['pregunta1_si'])
    mostrar_celda_grilla(axs[6, 2], 'pregunta1_no', celdas['pregunta1_no'])

    mostrar_celda_grilla(axs[7, 0], 'pregunta2', celdas['pregunta2'])
    mostrar_celda_grilla(axs[7, 1], 'pregunta2_si', celdas['pregunta2_si'])
    mostrar_celda_grilla(axs[7, 2], 'pregunta2_no', celdas['pregunta2_no'])
    
    mostrar_celda_grilla(axs[8, 0], 'pregunta3', celdas['pregunta3'])
    mostrar_celda_grilla(axs[8, 1], 'pregunta3_si', celdas['pregunta3_si'])
    mostrar_celda_grilla(axs[8, 2], 'pregunta3_no', celdas['pregunta3_no'])

    mostrar_celda_grilla(axs[9, 0], 'comentario', celdas['comentario'])
    ax_comentario_valor = plt.subplot2grid((10, 3), (9, 1), colspan=2)
    mostrar_celda_grilla(ax_comentario_valor, 'comentario_valor', celdas['comentario_valor'])

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.9)
    fig.suptitle('Formulario Desarmado por Celdas', fontsize=16)
    plt.show()
    
def binarizar_celda(celda_img):
    """Devuelve binaria con tinta=1 (blanco) y fondo=0 (negro)."""
    # Otsu sobre invertida para que la “tinta” quede alta
    # (si tu fondo es claro y la tinta oscura, invertimos así).
    _, th = cv2.threshold(255 - celda_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (th > 0).astype(np.uint8)
    
def extraer_componentes(celda, th_area_frac=0.002):
    """
    Etiqueta y filtra componentes pequeñas.
    th_area_frac es fracción del área de la celda para definir umbral mínimo.
    Devuelve (stats_filtrados_ordenados_x, centroids_filtrados).
    """
    h, w = celda.shape
    num, labels, stats, cents = cv2.connectedComponentsWithStats(celda, 8, cv2.CV_32S)

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

if __name__ == '__main__':
    mostrar_pasos = False
    figura_flag = False
    segmentos_flag = False
    vertices_flag = False
    celdas_flag = True
    
    formularios = ['formulario_01.png', 'formulario_02.png', 'formulario_03.png','formulario_04.png','formulario_05.png']
   
    for formulario in formularios:
        img, mascara, vert, hor = detectar_lineas(formulario, 180, 170, 200)
        segmentos_horizontales = encontrar_segmentos(mascara, hor, 'horizontal', 30)
        segmentos_verticales = encontrar_segmentos(mascara, vert, 'vertical', 30)
        vertices_form = encontrar_intersecciones(segmentos_horizontales, segmentos_verticales)
        celdas = encontrar_celdas(img, segmentos_horizontales, segmentos_verticales, margen=2)
        
        if (mostrar_pasos):
            img_para_dibujar = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) #type: ignore
            if (segmentos_flag):
                dibujar_segmentos_horizontales(img_para_dibujar, segmentos_horizontales, color=(255,0,0))
                dibujar_segmentos_verticales(img_para_dibujar, segmentos_verticales, color=(0,0,255))

            if (vertices_flag):
                dibujar_vertices(img_para_dibujar, vertices_form, color=(0,255,0), radio=2)
                
            if (figura_flag):
                plt.figure(figsize=(12, 12))
                plt.imshow(cv2.cvtColor(img_para_dibujar, cv2.COLOR_BGR2RGB))
                plt.title(f"Resultado de {formulario}")
                plt.axis('off')
                plt.show()
            
            if(celdas_flag):
                mostrar_formulario_desarmado(celdas)
        b = extraer_componentes(binarizar_celda(celdas['mail_valor']))
        palabras = contar_palabras_y_chars(b[0])
        print('----------------------------------')
        print(f'Formulario: {formulario}')
        print(f"Número de palabras en 'nombre_valor': {palabras[0]}")
        print(f"Número de caracteres en 'nombre_valor': {palabras[1]}")
        print(f"Cortes de palabras (índices de caracteres): {palabras}")
            
    
