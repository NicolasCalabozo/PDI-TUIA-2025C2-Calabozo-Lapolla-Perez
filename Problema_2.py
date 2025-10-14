import numpy as np
import matplotlib.pyplot as plt
import cv2


if __name__ == '__main__':
    img = cv2.imread(filename='formulario_vacio.png', flags=cv2.IMREAD_GRAYSCALE)
    img_para_dibujar = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) #type: ignore
    
    #Umbralado para separar el fondo devuelve una mascara binaria True|False del tamaño de la imagen
    #Resulta en una imagen con los colores invertidos, util para detectar lineas
    mascara_umbral = img < 160 #type: ignore
    
    #Sumamos los True en las columnas para generar un vector que nos diga
    #en qué coordenada X hay más concentracion de pixeles "blancos" (con valor 1)
    proyeccion_vertical = np.sum(mascara_umbral, axis=0)
    proyeccion_horizontal = np.sum(mascara_umbral, axis=1)
    
    #Seteamos un umbral vertical para que nos dé el indice del vector en donde la suma de pixeles es
    #mayor a 170 (probando a manopla los valores hasta que nos dé lo que queremos)
    umbral_vertical = 170
    #Hacemos lo mismo para las horizontales
    umbral_horizontal = 250
    coord_x_lineas = np.where(proyeccion_vertical > umbral_vertical)[0]
    coord_y_lineas = np.where(proyeccion_horizontal > umbral_horizontal)[0]

    #parametros para dibujar
    alto, ancho, _ = img_para_dibujar.shape
    color_rojo = (0, 0, 255)
    grosor_linea = 1

    #Graficamos las lineas horizontales
    for y in coord_y_lineas:
        #Inicio entre 0 y la altura de indice de y
        punto_inicio = (0, int(y))
        #Final entre el ancho máximo de la imagen y la altura de indice de y
        punto_fin = (ancho, int(y))
        cv2.line(img_para_dibujar, punto_inicio, punto_fin, color_rojo, grosor_linea)
    
    #Graficamos las lineas verticales
    for x in coord_x_lineas:
        punto_inicio = (int(x), 0)
        punto_fin = (int(x), alto)
        cv2.line(img_para_dibujar, punto_inicio, punto_fin, color_rojo, grosor_linea)

    plt.figure(figsize=(12, 12))
    plt.imshow(cv2.cvtColor(img_para_dibujar, cv2.COLOR_BGR2RGB))
    plt.title('Líneas Rojas Detectadas')
    plt.axis('off')
    plt.show()