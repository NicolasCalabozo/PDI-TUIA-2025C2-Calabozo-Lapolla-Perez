import numpy as np
import matplotlib.pyplot as plt
import cv2

def ecualizacionLocalHistograma(img, ventana_m: int, ventana_n: int):
    #Establecemos el tamaño de borde que vamos a necesitar para el padding
    borde_m = ventana_m // 2
    borde_n = ventana_n // 2
    #Introducimos el padding replicando el borde original
    img_borde = cv2.copyMakeBorder(img, borde_m, borde_m, borde_n, borde_n, borderType=cv2.BORDER_REPLICATE)    
    
    #Creamos una matriz del mismo tamaño que la imagen original
    img_salida = np.zeros_like(img)
    #Extraemos la ventana, ecualizamos y mapeamos el valor del pixel del centro
    #al pixel de salida correspondiente
    for y in range(borde_m,img_borde.shape[0]-borde_m):
        for x in range(borde_n,img_borde.shape[1]-borde_n):
            #El uso de +1 corresponde al funcionamiento del slicing en python.
            #de no usarlo estamos utilizando una ventana de m-1 x n-1
            roi = img_borde[y-borde_m:y+borde_m +1, x-borde_n:x+borde_n+1]
            roi_ecualizado = cv2.equalizeHist(roi) 
            img_salida[y-borde_m,x-borde_n] = roi_ecualizado[borde_m, borde_n]               
    return img_salida

def mostrar_analisis_ventana(lista_imagenes: list[np.ndarray], lista_titulos: list[str], titulo_principal: str):
    """
    Crea y muestra una grilla con un layout automático según la cantidad de imágenes.
    """
    n_imagenes = len(lista_imagenes)
    cols = int(np.ceil(np.sqrt(n_imagenes)))
    rows = int(np.ceil(n_imagenes / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig.suptitle(titulo_principal, fontsize=16)
    ax = ax.ravel() 
    
    for i in range(n_imagenes):
        ax[i].imshow(lista_imagenes[i], cmap='gray')
        ax[i].set_title(lista_titulos[i])
        ax[i].axis('off')

    for i in range(n_imagenes, rows * cols):
        ax[i].axis('off')

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()
    
if __name__ == '__main__':
    img = cv2.imread(filename='Imagen_con_detalles_escondidos.tif', flags=cv2.IMREAD_GRAYSCALE)
    tamaño_ventanas = [(5,5),(20,20),(50,50),(100,100), (150,150),(img.shape[0], img.shape[1])] #type:ignore
    imgs = []
    titulos = []
    for m, n in tamaño_ventanas:
        imgs.append(ecualizacionLocalHistograma(img,m,n))
        titulos.append(f'Ventana {m}x{n}')
    mostrar_analisis_ventana(imgs,titulos,'Análisis de Ecualización con Variación de Ventana')