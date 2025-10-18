import cv2
import numpy as np

def contar_espacios_y_palabras(celda):
    # Cambiamos el umbral de 180 a 140 porque estabamos perdiendo muchos pixeles de los puntos
    # No logrando la detección del punto como un componente
    _, binarizada = cv2.threshold(celda, 140, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binarizada, 8, cv2.CV_32S)  # type: ignore
    componentes = stats[1:]

    # Si no hay componentes tampoco hay espacios ni palabras
    if len(componentes) == 0:
        return 0, 0, 0

    # Ordenamos las componentes mediante el punto superior izquierdo
    componentes_ordenados = componentes[componentes[:,
                                                    cv2.CC_STAT_LEFT].argsort()]

    distancias = []

    umbral_dinamico = 0.
    cantidad_espacios = 0

    # Iteramos sobre los componentes para encontrar las distancias entre un caracter y el siguiente
    for i in range(len(componentes_ordenados) - 1):
        fin_actual = componentes_ordenados[i, cv2.CC_STAT_LEFT] + \
            componentes_ordenados[i, cv2.CC_STAT_WIDTH]
        inicio_siguiente = componentes_ordenados[i+1, cv2.CC_STAT_LEFT]
        distancia = inicio_siguiente - fin_actual
        if distancia > 0:
            distancias.append(distancia)

    # Si no tenemos distancias, es porque tenemos una sola palabra
    if not distancias:
        return num_labels-1, 1, 0

    # Nuestro caso especial es si tenemos dos distancias solamente, donde nuestro umbral dinamico no funcionaría
    if len(distancias) <= 2:
        # Para solucionarlo sacamos la media del ancho de los componentes
        ancho_promedio_caracter = np.mean(
            componentes_ordenados[:, cv2.CC_STAT_WIDTH])
        # Utilizamos un umbral para encontrar la cantidad de espacios con tamaño mayor
        # al 75% del promedio de los anchos de los componentes
        umbral_heuristico = ancho_promedio_caracter * 0.75
        cantidad_espacios = np.sum(distancias > umbral_heuristico)
    else:
        # Para tres componentes o más, utilizamos el MAD para encontrar
        # cuales distancias entre caracteres son atípicas y así encontrar la cantidad de espacios
        distancias = np.array(distancias)
        mediana = np.median(distancias)
        mad = np.median(np.abs(distancias - np.median(distancias)))
        if mad == 0:
            umbral_dinamico = mediana * 2.5
        else:
            umbral_dinamico = mediana + 3 * mad
        cantidad_espacios = np.sum(distancias > umbral_dinamico)

    # Llegado a este punto, sabemos que la cantidad de palabras es una más que la cantidad de espacios
    # Que nuestras componentes son todas menos el fondo, y la cantidad de espacios la calculada por nuestro método
    return num_labels-1, cantidad_espacios+1, cantidad_espacios

def validacion_nombre(celda):
    num_caracteres, num_palabras, num_espacios = contar_espacios_y_palabras(
        celda)
    if (num_palabras >= 2 and num_caracteres <= 25):
        return "OK"
    return "MAL"


def validacion_edad(celda):
    num_caracteres, num_palabras, num_espacios = contar_espacios_y_palabras(
        celda)
    if (2 <= num_caracteres <= 3 and num_espacios == 0):
        return "OK"
    return "MAL"


def validacion_mail(celda):
    num_caracteres, num_palabras, num_espacios = contar_espacios_y_palabras(
        celda)
    if (num_palabras == 1 and num_caracteres <= 25):
        return "OK"
    return "MAL"


def validacion_legajo(celda):
    num_caracteres, num_palabras, num_espacios = contar_espacios_y_palabras(
        celda)
    if (num_caracteres == 8 and num_palabras == 1):
        return "OK"
    return "MAL"


def validacion_comentario(celda):
    num_caracteres, num_palabras, num_espacios = contar_espacios_y_palabras(
        celda)
    if (num_palabras >= 1 and num_caracteres <= 25):
        return "OK"
    return "MAL"


def validacion_preguntas(celda_si, celda_no):
    num_caracteres_si, num_palabras_si, num_espacios_si = contar_espacios_y_palabras(
        celda_si)
    num_caracteres_no, num_palabras_no, num_espacios_no = contar_espacios_y_palabras(
        celda_no)

    if (num_caracteres_si == 1 and num_caracteres_no == 0):
        return "OK"

    if (num_caracteres_si == 0 and num_caracteres_no == 1):
        return "OK"

    return "MAL"


def validacion(celdas, id):
    '''
    Función que genera un diccionario con las validaciones de cada celda del formulario

    id: identificador del formulario
    tipo_formulario: [A,B]
    El resto de las variables asumen los valores: ['OK','MAL]
    '''
    estados = {}
    estados['id'] = id
    if id in ['01', '02', '03']:
        estados['tipo_formulario'] = 'A'
    else:
        estados['tipo_formulario'] = 'B'
    estados['nombre'] = validacion_nombre(celdas['nombre_valor'])
    estados['edad'] = validacion_edad(celdas['edad_valor'])
    estados['mail'] = validacion_mail(celdas['mail_valor'])
    estados['legajo'] = validacion_legajo(celdas['legajo_valor'])
    estados['pregunta1'] = validacion_preguntas(
        celdas['pregunta1_si'], celdas['pregunta1_no'])
    estados['pregunta2'] = validacion_preguntas(
        celdas['pregunta2_si'], celdas['pregunta2_no'])
    estados['pregunta3'] = validacion_preguntas(
        celdas['pregunta3_si'], celdas['pregunta3_no'])
    estados['comentario'] = validacion_comentario(celdas['comentario_valor'])
    return estados


def estado_formulario(estados):
    '''
    Función que encuentra el estado general del formulario
    Si uno de los campos es 'MAL', es considerado inválido
    '''
    aux = list(estados.values())
    for value in aux[2:]:
        if value != 'OK':
            return False
    return True