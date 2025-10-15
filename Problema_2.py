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
    print(x_unicas)
    print(y_unicas)
    #return lineas verticales, lineas horizontales
    return img, x_unicas, y_unicas

if __name__ == '__main__':
    img, vert, hor = detectar_lineas('formulario_vacio.png', 180, 170, 200)
    img_para_dibujar = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) #type: ignore

    # más ancha donde antes había dos.
    cajas_fusionadas = []
    for j in range(len(hor) - 1):
        for i in range(len(vert) - 1):
            y1, y2 = hor[j], hor[j+1]
            x1, x2 = vert[i], vert[i+1]
            caja_actual = {'top_left': (x1, y1), 'bottom_right': (x2, y2)}
            cajas_fusionadas.append(caja_actual)
    print(cajas_fusionadas)
    