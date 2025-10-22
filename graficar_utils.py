import matplotlib.pyplot as plt
import numpy as np
import cv2

def mostrar_celda_grilla(ax, titulo, imagen):
    ax.imshow(imagen, cmap='gray', vmin=0, vmax=255)
    ax.set_title(titulo)
    ax.axis('on')
    ax.set_xticks([])
    ax.set_yticks([])


def mostrar_formulario_desarmado(celdas):
    '''
    Función para mostrar el formulario separado por celdas, utiliza una función auxiliar para graficar
    '''
    fig, axs = plt.subplots(10, 3, figsize=(10, 15))

    for ax_row in axs:
        for ax in ax_row:
            ax.axis('off')

    mostrar_celda_grilla(axs[1, 0], 'nombre', celdas['nombre'])
    ax_nombre_valor = plt.subplot2grid((10, 3), (1, 1), colspan=2)
    mostrar_celda_grilla(ax_nombre_valor, 'nombre_valor',
                         celdas['nombre_valor'])

    mostrar_celda_grilla(axs[2, 0], 'edad', celdas['edad'])
    ax_edad_valor = plt.subplot2grid((10, 3), (2, 1), colspan=2)
    mostrar_celda_grilla(ax_edad_valor, 'edad_valor', celdas['edad_valor'])

    mostrar_celda_grilla(axs[3, 0], 'mail', celdas['mail'])
    ax_mail_valor = plt.subplot2grid((10, 3), (3, 1), colspan=2)
    mostrar_celda_grilla(ax_mail_valor, 'mail_valor', celdas['mail_valor'])
    mostrar_celda_grilla(axs[4, 0], 'legajo', celdas['legajo'])
    ax_legajo_valor = plt.subplot2grid((10, 3), (4, 1), colspan=2)
    mostrar_celda_grilla(ax_legajo_valor, 'legajo_valor',
                         celdas['legajo_valor'])
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
    mostrar_celda_grilla(ax_comentario_valor,
                         'comentario_valor', celdas['comentario_valor'])

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.9)
    fig.suptitle('Formulario Desarmado por Celdas', fontsize=16)
    plt.show()
    
def dibujar_segmentos_horizontales(imagen, segmentos_hor, color=(255, 0, 0), grosor=2):
    for y, segs in segmentos_hor.items():
        for x_inicio, x_fin in segs:
            cv2.line(imagen, (x_inicio, y), (x_fin, y), color, grosor)


def dibujar_segmentos_verticales(imagen, segmentos_ver, color=(0, 0, 255), grosor=2):
    for x, segs in segmentos_ver.items():
        for y_inicio, y_fin in segs:
            cv2.line(imagen, (x, y_inicio), (x, y_fin), color, grosor)
            
def graficar_estado_formulario(lista_celdas_nombre, lista_estados_generales):
    '''
    Función para graficar los estados generales de cada formulario.
    Marker 'O' en caso de que el formulario esté 'OK' en todas las celdas
    Marker 'X' en caso de que el formulario tenga al menos un 'MAL' en una de las celdas
    '''
    num_formularios = len(lista_celdas_nombre)

    fig, axes = plt.subplots(
        num_formularios, 1, figsize=(num_formularios * 4, 4))

    if num_formularios == 1:
        axes = [axes]

    for ax, celda, estado in zip(axes, lista_celdas_nombre, lista_estados_generales):

        ax.imshow(celda, cmap='gray', vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        
        #Posicionamiento relativo de los markers 'O' y 'X'
        h, w = celda.shape[:2]
        x_pos = w * 0.6  
        y_pos = h * 0.3   
        if estado:
            # Ponemos un círculo verde
            ax.text(x_pos, y_pos, 'O',
                    color='green',
                    ha='right',
                    va='top',
                    fontweight='bold',
                    fontsize=25)
        else:
            # Ponemos una X roja
            ax.text(x_pos, y_pos, 'X',
                    color='red',
                    ha='right',
                    va='top',
                    fontweight='bold',
                    fontsize=25)

    plt.suptitle('Resultados de Validación (Apartado C)', fontsize=16)
    plt.tight_layout()
    plt.show()
    
def generar_imagen_dummy(ancho=300, alto=100, texto="Formulario C"):
    '''
    Función que crea una imagen dummy para ver si nuestra función detecta Formularios tipo C
    '''
    imagen_dummy = np.zeros((alto, ancho), dtype=np.uint8)
    cv2.putText(imagen_dummy, texto, (10, int(alto * 0.7) ),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
    return imagen_dummy