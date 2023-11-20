# Para capturar el fotograma
import cv2

# Para procesar la matriz de la imagen
import numpy as np


# Importar el módulo tensorflow y cargar el modelo

import tensorflow as tf

modelo = tf.keras.models.load_model('keras_model.h5')

# Adjuntando el índice de la cámara como 0 con la aplicación del software
camera = cv2.VideoCapture(0)

# Bucle infinito
while True:

	# Leyendo/Solicitando un fotograma de la cámara
	status , frame = camera.read()

	# Si somos capaces de leer exitosamente el fotograma
	if status:

		# Voltear la imagen
		frame = cv2.flip(frame , 1)
		
		# Redimensionar el fotograma
		
		imagen = cv2.resize(frame, (224,224))

		# Expandir las dimensiones 
		
		imagen_test = np.array(imagen, dtype=np.float32)
		imagen_test = np.expand_dims(imagen_test, axis=0)

		# Normalizar antes de alimentar al modelo
		
		imagen_normalizada = imagen_test/255.0

		# Obtener predicciones del modelo
		
		prediccion = modelo.predict(imagen_normalizada)

		print("Predicción: ", prediccion)
		
		# Mostrando los fotogramas capturados
		cv2.imshow('Alimentar' , frame)

		# Esperando 1ms
		code = cv2.waitKey(1)
		
		# Si se preciona la barra espaciadora, romper el bucle
		if code == 32:
			break

# Liberar la cámara de la aplicación del software
camera.release()

# Cerrar la ventana abierta
cv2.destroyAllWindows()
