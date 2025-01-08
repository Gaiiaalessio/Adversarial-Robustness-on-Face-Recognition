import numpy as np
from PIL import Image

# Carica i file .npy
data_1 = np.load('expdemo/1.npy')
image = Image.open('../../../data/lfw-112x112-new/Abel_Pacheco/Abel_Pacheco_0004.jpg')

# Converte l'immagine in un array Numpy
image_array = np.array(image)

# Visualizza informazioni sui dati
print("Informazioni su 1.npy:")
print(f"Forma: {data_1.shape}")
print(f"Minimo: {data_1.min()}")
print(f"Massimo: {data_1.max()}")
print(f"Media: {data_1.mean()}")

print("\nInformazioni su data_1:")
print(f"Forma: {image_array.shape}")
print(f"Minimo: {image_array.min()}")
print(f"Massimo: {image_array.max()}")
print(f"Media: {image_array.mean()}")

# Calcola la distanza L2 e L∞
l2_distance = np.sqrt(np.sum((data_1 - image_array) ** 2)) / np.sqrt(data_1.size)
l_inf_distance = np.max(np.abs(data_1 - image_array))

print(f"Distanza L2: {l2_distance}")
print(f"Distanza L∞: {l_inf_distance}")

import matplotlib.pyplot as plt

plt.subplot(1, 2, 1)
plt.imshow(image_array)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(data_1.astype(np.uint8))  # Converte a uint8 per mostrare come immagine
plt.title("Adversarial Image")
plt.axis('off')

plt.show()


