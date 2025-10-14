import numpy as np
import matplotlib.pyplot as plt
import cv2

def ecualizacionLocalHistograma(img, ventana_m: int, ventana_n: int, debug=False):
    borde_m = ventana_m // 2
    borde_n = ventana_n // 2
    img_borde = cv2.copyMakeBorder(img, borde_m, borde_m, borde_n, borde_n, borderType=cv2.BORDER_REPLICATE)
    
    if debug:
        print(img_borde)    
    
    img_salida = np.zeros_like(img)
    for y in range(borde_m,img_borde.shape[0]-borde_m):
        for x in range(borde_n,img_borde.shape[1]-borde_n):
            roi = img_borde[y-borde_m:y+borde_m, x-borde_n:x+borde_n]
            roi_ecualizado = cv2.equalizeHist(roi) 
            img_salida[y-borde_m,x-borde_n] = roi_ecualizado[borde_m, borde_n]               
    return img_salida

if __name__ == '__main__':
    img = cv2.imread(filename='Imagen_con_detalles_escondidos.tif', flags=cv2.IMREAD_GRAYSCALE)
    img_salida = ecualizacionLocalHistograma(img, 20, 20, True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].set_title('Imagen Original')
    ax[0].imshow(img, cmap='gray')
    ax[0].axis('off')
    ax[1].set_title('Ecualización Local')
    ax[1].imshow(img_salida, cmap='gray')
    ax[1].axis('off')
    plt.show()