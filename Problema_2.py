import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
from graficar_utils import dibujar_segmentos_horizontales,dibujar_segmentos_verticales,graficar_estado_formulario,mostrar_formulario_desarmado
from validar_utils import validacion, estado_formulario

def encontrar_lineas(nombre_archivo, umbral_mascara, umbral_vert, umbral_hor):
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
    celdas['comentario_valor'] = filas[9][:,
                                          seg_ver[1]+margen: seg_ver[3]-margen]

    return celdas


def escribir_csv(estados):
    with open(file='estados_formularios.csv', mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id','nombre_y_apellido','edad','mail','legajo','pregunta1','pregunta2','pregunta3','comentarios'])
        for _, estado in estados.items():
            writer.writerow([estado['id'], estado['nombre'], estado['edad'], estado['mail'], estado['legajo'],
                            estado['pregunta1'], estado['pregunta2'], estado['pregunta3'], estado['comentario']])


if __name__ == '__main__':
    mostrar_pasos = False; figura_flag = False; segmentos_flag = False; celdas_flag = False
    img_salida_flag = True; crear_csv = False
    id_formularios = []; celdas_nombre = []; lista_estados_generales = []; lista_celdas_nombre = []
    estados = {}; estados_completos = {}
    formularios = ['formulario_01.png', 'formulario_02.png','formulario_03.png', 'formulario_04.png', 'formulario_05.png']
    for formulario in formularios:
        id_formulario = formulario.split(sep="_")[1][:2]
        id_formularios.append(id_formulario)
        img, mascara, vert, hor = encontrar_lineas(formulario, 180, 170, 200)
        segmentos_horizontales = encontrar_segmentos(mascara, hor, 'horizontal', 30)
        segmentos_verticales = encontrar_segmentos(mascara, vert, 'vertical', 30)
        celdas = encontrar_celdas(img, segmentos_horizontales, segmentos_verticales, margen=2)
        lista_celdas_nombre.append(celdas['nombre_valor'])
        estado = validacion(celdas, id_formulario)
        estados[id_formulario] = estado
        estado_general = estado_formulario(estado)
        lista_estados_generales.append(estado_general)

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

    if img_salida_flag:
        graficar_estado_formulario(
            lista_celdas_nombre, lista_estados_generales)
        
    if crear_csv:
        escribir_csv(estados)