import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
from graficar_utils import dibujar_segmentos_horizontales,dibujar_segmentos_verticales,graficar_estado_formulario,mostrar_formulario_desarmado, generar_imagen_dummy
from validar_utils import validacion, estado_formulario

def encontrar_lineas(nombre_archivo, umbral_mascara, longitud_min_vertical, longitud_min_horizontal):
    '''
    Función para encontrar coordenadas de lineas horizontales y verticales
    
    Argumentos necesarios:
    nombre_archivo: Ruta al archivo,
    umbral_mascara: Umbral general para binarizar la imagen,
    longitud_min_vertical: Valor umbral relacionado a la cantidad de pixeles necesarios para ser considerado una linea vertical,
    longitud_min_horizontal: Valor umbral relacionado a la cantidad de píxeles necesarios para ser considerados una linea horizontal
    
    Retorna:
    img: imagen original,
    mascara_umbral: imagen original binarizada,
    x_unicas: coordenadas de x sobre las cuales tenemos lineas verticales,
    y_unicas: coordenadas de y sobre las cuales tenemos lineas horizontales
    '''
    img = cv2.imread(filename=nombre_archivo, flags=cv2.IMREAD_GRAYSCALE)
    mascara_umbral = img < umbral_mascara  # type: ignore
    # Sumamos los True para generar un vector que nos diga
    # en qué coordenada X o Y hay más concentracion de pixeles
    proyeccion_vertical = np.sum(mascara_umbral, axis=0)
    proyeccion_horizontal = np.sum(mascara_umbral, axis=1)
    coord_x_lineas = list(np.where(proyeccion_vertical > longitud_min_vertical)[0])
    coord_y_lineas = list(np.where(proyeccion_horizontal > longitud_min_horizontal)[0])
    #Como nuestro umbralado general genera lineas dobles, descartamos una linea intermedia para quedarnos con una sola
    x_unicas = coord_x_lineas[1::2]
    y_unicas = coord_y_lineas[1::2]
    return img, mascara_umbral, x_unicas, y_unicas


def encontrar_segmentos(mascara_umbral, coordenadas, eje='horizontal', min_largo=10):
    '''
    Función que encuentra los segmentos de linea teniendo en cuenta el eje y largo mínimo,
    utilizada para poder graficar las separaciones del formulario
    
    Argumentos necesarios:
    mascara_umbral: Imagen binarizada,
    coordenadas: Vector que contiene las coordenadas X o Y donde encontraremos potenciales segmentos horizontales o verticales
    eje: 'horizontal' o 'vertical' para encontrar segmentos con la orientación deseada,
    min_largo: valor umbral que determina el largo minimo (pixeles consecutivos) para ser considerado un segmento.
    
    Retorna:
    segmentos_dict: Diccionario con clave "coordenada_base" y valor [(inicio, fin)...] para identificar la columna,
    el comienzo y el final de los segmentos.
    Ej: {50: [(10, 100), (120, 200)]} En la fila Y=50, hay un segmento de X=10 a X=100 y otro de X=120 a X=200.
    '''
    
    segmentos_dict = {}
    linea_actual = []
    for coord in coordenadas:
        #Extraemos la linea donde encontraremos los segmentos
        #Si es horizontal extraemos una fila. Si es vertical, una columna.
        if eje == 'horizontal':
            linea_actual = mascara_umbral[coord, :]
        elif eje == 'vertical':
            linea_actual = mascara_umbral[:, coord]
        segmentos_en_linea = []
        en_segmento = False
        inicio = 0

        #Contamos la cantidad de pixeles negros consecutivos, si superan el largo minimo, se consideran segmentos
        for i, pixel_es_negro in enumerate(linea_actual):
            if pixel_es_negro and not en_segmento:
                en_segmento = True
                inicio = i
            elif not pixel_es_negro and en_segmento:
                en_segmento = False
                final = i - 1
                if (final - inicio) >= min_largo:
                    segmentos_en_linea.append((inicio, final))
        
        #Si el segmento toma todo el ancho o el alto de la imagen, en_segmento se mantiene en True
        if en_segmento:
            final = len(linea_actual) - 1
            if (final - inicio) >= min_largo:
                segmentos_en_linea.append((inicio, final))

        if segmentos_en_linea:
            segmentos_dict[coord] = segmentos_en_linea

    return segmentos_dict

def encontrar_celdas(img, segmentos_hor, segmentos_ver, margen=2):
    '''
    Función que separa las celdas individuales del formulario
    
    Argumentos necesarios:
    img: imagen original,
    segmentos_hor: Diccionario con los segmentos horizontales,
    segmentos_ver: Diccionario con los segmentos verticales,
    margen: Valor que permite hacer un cropping de la imagen hacia adentro para eliminar bordes residuales
    
    Retorna:
    celdas: Diccionario con clave 'nombre_celda' y valor 'seccion de la imagen que representa la celda'
    '''
    filas = []
    #Separamos la imagen en filas, recortando pixeles equivalentes al margen de la parte superior e inferior de la fila
    seg_hor = sorted(list(segmentos_hor.keys()))
    for i in range(len(seg_hor) - 1):
        #Hay que tener cuidado con el slicing de python, se agrega un +1 para poder recortar efectivamente 
        #el valor del margen
        fila_recortada = img[seg_hor[i]+1+margen: seg_hor[i+1] - margen, :]
        filas.append(fila_recortada)

    seg_ver = sorted(list(segmentos_ver.keys()))
    celdas = {}
    #Una vez obtenidas las filas, separamos por columnas según el campo
    #recortando pixeles equivalentes al margen del lateral izquierdo y derecho de la celda
    celdas['titulo'] = filas[0][:, seg_ver[0]+1+margen: seg_ver[3]-margen]

    celdas['nombre'] = filas[1][:, seg_ver[0]+1+margen: seg_ver[1]-margen]
    celdas['nombre_valor'] = filas[1][:, seg_ver[1]+1+margen: seg_ver[3]-margen]

    celdas['edad'] = filas[2][:, seg_ver[0]+1+margen: seg_ver[1]-margen]
    celdas['edad_valor'] = filas[2][:, seg_ver[1]+1+margen: seg_ver[3]-margen]

    celdas['mail'] = filas[3][:, seg_ver[0]+1+margen: seg_ver[1]-margen]
    celdas['mail_valor'] = filas[3][:, seg_ver[1]+1+margen: seg_ver[3]-margen]

    celdas['legajo'] = filas[4][:, seg_ver[0]+1+margen: seg_ver[1]-margen]
    celdas['legajo_valor'] = filas[4][:, seg_ver[1]+1+margen: seg_ver[3]-margen]

    celdas['pregunta1'] = filas[6][:, seg_ver[0]+1+margen: seg_ver[1]-margen]
    celdas['pregunta1_si'] = filas[6][:, seg_ver[1]+1+margen: seg_ver[2]-margen]
    celdas['pregunta1_no'] = filas[6][:, seg_ver[2]+1+margen: seg_ver[3]-margen]

    celdas['pregunta2'] = filas[7][:, seg_ver[0]+1+margen: seg_ver[1]-margen]
    celdas['pregunta2_si'] = filas[7][:, seg_ver[1]+1+margen: seg_ver[2]-margen]
    celdas['pregunta2_no'] = filas[7][:, seg_ver[2]+1+margen: seg_ver[3]-margen]

    celdas['pregunta3'] = filas[8][:, seg_ver[0]+1+margen: seg_ver[1]-margen]
    celdas['pregunta3_si'] = filas[8][:, seg_ver[1]+1+margen: seg_ver[2]-margen]
    celdas['pregunta3_no'] = filas[8][:, seg_ver[2]+1+margen: seg_ver[3]-margen]

    celdas['comentario'] = filas[9][:, seg_ver[0]+1+margen: seg_ver[1]-margen]
    celdas['comentario_valor'] = filas[9][:,
                                          seg_ver[1]+1+margen: seg_ver[3]-margen]

    return celdas

def determinar_tipo_formulario(celda):
    '''
    Función que toma la celda correspondiente a la primer fila del formulario y retorna el tipo de formulario A,B o C
    '''
    #Necesitamos una celda de fondo negro y letras blancas para encontrar contornos, asique invertimos la imagen
    celda_invertida = cv2.bitwise_not(celda)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(celda_invertida, connectivity=8)
    #Creamos una máscara de ceros
    mascara_letra = np.zeros_like(celda_invertida, dtype=np.uint8)
    #Eliminamos el fondo
    component_stats = stats[1:]
    #Encontramos la componente más a la derecha
    #Como eliminamos el fondo, la componente más a la derecha será la letra "A", "B" o "C"
    indice_mas_derecha = np.argmax(component_stats[:, cv2.CC_STAT_LEFT])
    #Le sumamos uno teniendo en cuenta la eliminación del fondo
    label_interes = indice_mas_derecha + 1
    mascara_letra[labels == label_interes] = 255
    #Encontramos los contornos
    contornos, jerarquia = cv2.findContours(mascara_letra, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    num_contornos_totales = len(contornos)
    #Definimos que letra es por la cantidad de contornos encontrados
    if num_contornos_totales == 3:
        return "B"
    elif num_contornos_totales == 2:
        return "A"
    elif num_contornos_totales == 1:
        return "C"

def escribir_csv(estados):
    '''
    Función para la creación/sobreescritura de los estados de cada formulario
    '''
    with open(file='estados_formularios.csv', mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id','nombre_y_apellido','edad','mail','legajo','pregunta1','pregunta2','pregunta3','comentarios'])
        for _, estado in estados.items():
            writer.writerow([estado['id'], estado['nombre'], estado['edad'], estado['mail'], estado['legajo'],
                            estado['pregunta1'], estado['pregunta2'], estado['pregunta3'], estado['comentario']])


if __name__ == '__main__':
    #-------------------------------- Funcionamiento Principal -------------------------------------
    #1. Setear en True las banderas 'mostrar_pasos', 'segmentos_flag', 'figura_flag' para observar
    #los segmentos utilizados luego para hacer el despiece en celdas del formulario
    
    #2. Setear en True las banderas 'mostrar_pasos' y 'celdas_flag' para observar el despiece por celdas
    #de cada formulario.
    
    #3. Setear en True la bandera 'img_salida_flag' para observar el estado general
    #de cada formulario según el nombre provisto.
    
    #4. Setear en True la bandera 'crear_csv' para crear/sobreescribir el archivo .csv con resumen de cada formulario.
    
    #5. Setear en True la bandera 'formulario_flag' si se desea ver el estado de cada formulario por consola.
    
    #6. Setear en true la bandera 'test_formulario_c' si desea probar el caso de que un formulario sea tipo C
    
    #Recomendamos tener activado 'formulario_flag' y 'crear_csv' en todo momento ya que responden al objetivo completo
    #del trabajo práctico
    #------------------------------------------------------------------------------------------------
    mostrar_pasos = True; figura_flag = False; segmentos_flag = False; celdas_flag = False
    img_salida_flag = False; crear_csv = True; formulario_flag = True; test_formulario_c = False
    id_formularios = []; celdas_nombre = []; lista_estados_generales = []; lista_celdas_nombre = []
    estados = {}; estados_completos = {}
    formularios = ['formulario_01.png', 'formulario_02.png','formulario_03.png', 'formulario_04.png', 'formulario_05.png']
    for formulario in formularios:
        #Obtención del ID del formulario
        id_formulario = formulario.split(sep="_")[1][:2]
        id_formularios.append(id_formulario)
        
        #Obtención de lineas
        img, mascara, vert, hor = encontrar_lineas(formulario, 180, 170, 200)
        
        #Obtención de segmentos
        segmentos_horizontales = encontrar_segmentos(mascara, hor, 'horizontal', 30)
        segmentos_verticales = encontrar_segmentos(mascara, vert, 'vertical', 30)
        
        #Obtención de celdas
        celdas = encontrar_celdas(img, segmentos_horizontales, segmentos_verticales, margen=2)
        
        #Guardamos la celda de los nombres para luego mostrar el estado general de cada formulario
        lista_celdas_nombre.append(celdas['nombre_valor'])
        
        #Determinamos el tipo de formulario
        tipo_formulario = determinar_tipo_formulario(celdas['titulo'])
        
        #Validamos cada celda
        estado = validacion(celdas, id_formulario, tipo_formulario)
        
        #Guardamos los estados de todos los formularios para generar un overview final
        estados[id_formulario] = estado
        estado_general = estado_formulario(estado)
        lista_estados_generales.append(estado_general)

        #Seccion para mostrar pasos intermedios
        if (mostrar_pasos):

            img_para_dibujar = cv2.cvtColor(
                img, cv2.COLOR_GRAY2BGR)  # type: ignore
            if (segmentos_flag):
                dibujar_segmentos_horizontales(
                    img_para_dibujar, segmentos_horizontales, color=(255, 0, 0))
                dibujar_segmentos_verticales(
                    img_para_dibujar, segmentos_verticales, color=(0, 0, 255))

            if (figura_flag):
                plt.figure(figsize=(12, 12))
                plt.imshow(cv2.cvtColor(img_para_dibujar, cv2.COLOR_BGR2RGB))
                plt.title(f"Resultado de {formulario}")
                plt.axis('off')
                plt.show()

           
            if (celdas_flag):
                mostrar_formulario_desarmado(celdas)
    
    
    if (img_salida_flag):
        graficar_estado_formulario(lista_celdas_nombre, lista_estados_generales)

    
    if (crear_csv):
        escribir_csv(estados)
    
    if (formulario_flag):
        print('----------------------')
        print('Estados de todos los formularios')
        for clave, estado in estados.items():
            print('----------------------')
            for campo, valor in estado.items():
                print(f'{campo}: {valor}')
            print(f'¿Es un formulario válido?: {estado_formulario(estado)}')
        print('----------------------\n')
        
    if (test_formulario_c):
        img_dummy = generar_imagen_dummy()
        print('----------------------')
        print(f'El formulario dummy es de Tipo {determinar_tipo_formulario(img_dummy)}')
        print('----------------------')
        plt.imshow(img_dummy, cmap='gray')
        plt.show()
