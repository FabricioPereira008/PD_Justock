import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'produto.jpg'  
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(10, 6))
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Imagem Original')
plt.axis('off')

_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
plt.subplot(2, 3, 2)
plt.imshow(binary_image, cmap='gray')
plt.title('Imagem Binária')
plt.axis('off')

kernel = np.ones((5, 5), np.uint8)
dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
plt.subplot(2, 3, 3)
plt.imshow(dilated_image, cmap='gray')
plt.title('Imagem Dilatada')
plt.axis('off')

equalized_image = cv2.equalizeHist(image)
plt.subplot(2, 3, 4)
plt.imshow(equalized_image, cmap='gray')
plt.title('Imagem com Realce de Contraste')
plt.axis('off')

edges = cv2.Canny(dilated_image, 100, 200)
plt.subplot(2, 3, 5)
plt.imshow(edges, cmap='gray')
plt.title('Bordas com Canny')
plt.axis('off')

plt.tight_layout()
plt.show()
