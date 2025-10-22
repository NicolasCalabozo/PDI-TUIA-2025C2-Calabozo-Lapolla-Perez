# IA 4.4 Procesamiento de Imágenes - Trabajo Práctico N°1
 Facultad de Ciencias Exactas, Ingeniería y Agrimensura
 
 Tecnicatura Universitaria en Inteligencia Artificial

## Integrantes

* Calabozo, Nicolás
* Lapolla, Martín
* Perez, Sebastián

Este repositorio contiene la resolución del Trabajo Práctico N°1 de Procesamiento de Imágenes I, el cual esta enfocado en técnicas de ecualización local de histograma y validación automática de formularios mediante procesamiento de imágenes.

## Estructura del Repositorio

* `Problema_1.py`: Script principal para la resolución del Problema 1 (Ecualización Local).
* `Problema_2.py`: Script principal para la resolución del Problema 2 (Validación de Formularios).
* `validar_utils.py`: Módulo con funciones auxiliares para la validación de campos del Problema 2.
* `graficar_utils.py`: Módulo con funciones auxiliares para graficar resultados del Problema 2.
* `Imagen_con_detalles_escondidos.tif`: Imagen de entrada para el Problema 1.
* `formulario_vacio.png`: Imagen del esquema del formulario (Problema 2).
* `formulario_XX.png`: Imágenes de los formularios completados (01 a 05) para el Problema 2.
* `README.md`: Este archivo.
* `requirements.txt`: Lista todas las librerias externas requeridas por python para ejecutar el trabajo

## Requisitos Previos

* Python 3.x
* OpenCV (`opencv-python`)
* NumPy (`numpy`)
* Matplotlib (`matplotlib`)

## Instalación

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/NicolasCalabozo/PDI-TUIA-2025C2-Calabozo-Lapolla-Perez
    cd PDI-TUIA-2025C2-Calabozo-Lapolla-Perez
    ```
2.  **Crear un entorno virtual (recomendado):**
    ```bash
    python -m venv env
    # En Windows
    .\env\Scripts\activate
    # En Linux
    source env/bin/activate
    ```
3.  **Instalar las dependencias:**
    ```bash
    `pip install -r requirements.txt`
    ```

## Instrucciones de Uso

Asegurarse de tener todas las imágenes necesarias (`.tif`, `.png`) en el mismo directorio que los scripts o en la ruta esperada por el código.

### Problema 1: Ecualización Local

Para ejecutar el script que revela los objetos ocultos y analiza el efecto del tamaño de ventana:

```bash
python Problema_1.py
```

### Problema 2: Validación de Formularios

Para ejecutar el script que **valida** los 5 formularios (`formulario_01.png` a `formulario_05.png`), genera el archivo CSV y una imagen que informa aquellas personas que han 
completado o no correctamente el formulario:

```bash
python Problema_2.py
