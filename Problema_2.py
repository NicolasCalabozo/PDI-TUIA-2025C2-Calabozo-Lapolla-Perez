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
    

def contar_espacios_y_palabras(celda):
    #Cambiamos el umbral de 180 a 140 porque estabamos perdiendo muchos pixeles de los puntos
    #No logrando la detección del punto como un componente
    _, binarizada = cv2.threshold(celda, 140, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binarizada, 8, cv2.CV_32S) #type: ignore
    componentes = stats[1:]
    
    if len(componentes) == 0:
        return 0, 0, 0
    
    componentes_ordenados = componentes[componentes[:, cv2.CC_STAT_LEFT].argsort()]

    distancias = []

    umbral_dinamico = 0.
    cantidad_espacios = 0
    for i in range(len(componentes_ordenados) - 1):
        fin_actual = componentes_ordenados[i, cv2.CC_STAT_LEFT] + componentes_ordenados[i, cv2.CC_STAT_WIDTH]
        inicio_siguiente = componentes_ordenados[i+1, cv2.CC_STAT_LEFT]
        distancia = inicio_siguiente - fin_actual
        if distancia > 0:
            distancias.append(distancia)
            
    if not distancias:
        return num_labels-1, 1, 0
    
    if len(distancias) <= 2: 
        ancho_promedio_caracter = np.mean(componentes_ordenados[:, cv2.CC_STAT_WIDTH])
        umbral_heuristico = ancho_promedio_caracter * 0.75
        cantidad_espacios = np.sum(distancias > umbral_heuristico)   
    else:
        distancias = np.array(distancias)
        mediana = np.median(distancias)
        mad = np.median(np.abs(distancias - np.median(distancias)))
        if mad == 0:
            umbral_dinamico = mediana * 2.5
        else:
            umbral_dinamico = mediana + 3 * mad
        cantidad_espacios = np.sum(distancias > umbral_dinamico)
    
    return num_labels-1, cantidad_espacios+1, cantidad_espacios

def validacion_nombre(celda):
    num_caracteres, num_palabras, num_espacios = contar_espacios_y_palabras(celda)
    if (num_palabras >= 2 and num_caracteres <=25):
        return "OK"
    return "MAL"
    
def validacion_edad(celda):
    num_caracteres, num_palabras, num_espacios = contar_espacios_y_palabras(celda)
    if (2 <= num_caracteres <=3 and num_espacios == 0):
        return "OK"
    return "MAL"

def validacion_mail(celda):
    num_caracteres, num_palabras, num_espacios = contar_espacios_y_palabras(celda)
    if (num_palabras == 1 and num_caracteres <=25):
        return "OK"
    return "MAL"

def validacion_legajo(celda):
    num_caracteres, num_palabras, num_espacios = contar_espacios_y_palabras(celda)
    if (num_caracteres == 8 and num_palabras == 1):
        return "OK"
    return "MAL"

def validacion_comentario(celda):
    num_caracteres, num_palabras, num_espacios = contar_espacios_y_palabras(celda)
    if (num_palabras >= 1 and num_caracteres <= 25):
        return "OK"
    return "MAL"

def validacion_preguntas(celda_si, celda_no):
    num_caracteres_si, num_palabras_si, num_espacios_si = contar_espacios_y_palabras(celda_si)
    num_caracteres_no, num_palabras_no, num_espacios_no = contar_espacios_y_palabras(celda_no)
    
    if (num_caracteres_si == 1 and num_caracteres_no == 0):
        return "OK"
    
    if (num_caracteres_si == 0 and num_caracteres_no == 1):
        return "OK"
    
    return "MAL"

def validacion(celdas, id):
    estados = {}
    estados['id'] = id
    if id in ['01','02','03']:
        estados['tipo_formulario'] = 'A'
    else:
        estados ['tipo_formulario'] = 'B'
    estados['nombre'] = validacion_nombre(celdas['nombre_valor'])
    estados['edad'] = validacion_edad(celdas['edad_valor'])
    estados['mail'] = validacion_mail(celdas['mail_valor'])
    estados['legajo']= validacion_legajo(celdas['legajo_valor'])
    estados['pregunta1'] = validacion_preguntas(celdas['pregunta1_si'], celdas['pregunta1_no'])
    estados['pregunta2'] = validacion_preguntas(celdas['pregunta2_si'], celdas['pregunta2_no'])
    estados['pregunta3'] = validacion_preguntas(celdas['pregunta3_si'], celdas['pregunta3_no'])
    estados['comentario'] = validacion_comentario(celdas['comentario_valor'])
    return estados

def estado_formulario(estados):
    aux = list(estados.values())
    for value in aux[2:]:
        if value != 'OK':
            return False
    return True

def graficar_estado_formulario(lista_celdas_nombre, lista_estados_generales):
    """
    Crea una figura que muestra la celda "Nombre y Apellido" de cada formulario
    con un indicador de texto (Círculo verde para OK, X roja para MAL) usando Matplotlib.
    
    Args:
        lista_celdas_nombre (list): Lista de las imágenes (arrays) de las celdas 'nombre_valor'.
        lista_estados_generales (list): Lista de strings ('OK' o 'MAL') con el estado de cada formulario.
    """
    
    num_formularios = len(lista_celdas_nombre)
    
    # 1. Creamos una grilla de subplots, uno para cada formulario
    fig, axes = plt.subplots(num_formularios, 1, figsize=(num_formularios * 4, 4))
    
    # Si solo hay un formulario, 'axes' no es un array, lo convertimos
    if num_formularios == 1:
        axes = [axes]
        
    # 2. Iteramos sobre los ejes, las celdas y sus estados
    for ax, celda, estado in zip(axes, lista_celdas_nombre, lista_estados_generales):
        
        # 3. Mostramos la imagen de la celda en escala de grises
        ax.imshow(celda, cmap='gray', vmin = 0, vmax = 255)
        ax.axis('off') # Ocultamos los ejes

        # 4. Definimos dónde poner el indicador (esquina superior derecha)
        h, w = celda.shape[:2]
        x_pos = w * 0.6  # 95% a la derecha
        y_pos = h * 0.1   # 10% desde arriba
        #print(f"El estado es: {estado}")
        # 5. Dibujamos el indicador (Círculo o X) usando ax.text
        if estado:
            # Ponemos un círculo verde
            ax.text(x_pos, y_pos, 'O', 
                    color='green', 
                    ha='right', 
                    va='top', 
                    fontweight='bold', 
                    fontsize=15)
        else:
            # Ponemos una X roja
            ax.text(x_pos, y_pos, 'X', 
                    color='red', 
                    ha='right', 
                    va='top', 
                    fontweight='bold', 
                    fontsize=15)

    plt.suptitle('Resultados de Validación (Apartado C)', fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    mostrar_pasos = False
    figura_flag =False
    segmentos_flag = False
    vertices_flag = False
    celdas_flag = False
    img_salida_flag = True
    estados = {}
    id_formularios = []
    celdas_nombre = []
    lista_estados_generales = []
    estados_completos = {}
    lista_celdas_nombre = []
    formularios = ['formulario_01.png', 'formulario_02.png', 'formulario_03.png','formulario_04.png','formulario_05.png']
    for formulario in formularios:
        id_formulario = formulario.split(sep="_")[1][:2]
        id_formularios.append(id_formulario)
        img, mascara, vert, hor = detectar_lineas(formulario, 180, 170, 200)
        segmentos_horizontales = encontrar_segmentos(mascara, hor, 'horizontal', 30)
        segmentos_verticales = encontrar_segmentos(mascara, vert, 'vertical', 30)
        vertices_form = encontrar_intersecciones(segmentos_horizontales, segmentos_verticales)
        celdas = encontrar_celdas(img, segmentos_horizontales, segmentos_verticales, margen=2)
        estados[id_formulario] = validacion(celdas, id_formulario)
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
                
        
        lista_celdas_nombre.append(celdas['nombre_valor'])
        estado_general = estado_formulario(estados[id_formulario])
        print(estado_general)
        lista_estados_generales.append(estado_general)
    if img_salida_flag:
        graficar_estado_formulario(lista_celdas_nombre, lista_estados_generales)

