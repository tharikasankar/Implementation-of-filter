# Implementation-of-filter
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1
Import the required libraries.


### Step2
Convert the image from BGR to RGB.


### Step3
Apply the required filters for the image separately.


### Step4
Plot the original and filtered image by using matplotlib.pyplot.


### Step5
End the program.
 

## Program:
### Developed By   : Tharika S
### Register Number 212222230159
</br>

### 1. Smoothing Filters

i) Using Averaging Filter
```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('fish.png', cv2.IMREAD_GRAYSCALE)
salt_prob = 0.05
pepper_prob = 0.05
noisy_image = np.copy(image)
num_salt = np.ceil(salt_prob * image.size)
coords_salt = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
noisy_image[tuple(coords_salt)] = 255
num_pepper = np.ceil(pepper_prob * image.size)
coords_pepper = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
noisy_image[tuple(coords_pepper)] = 0
filtered_image = np.zeros_like(noisy_image) 
height, width = noisy_image.shape
for i in range(1, height - 1):
    for j in range(1, width - 1):
        neighborhood = noisy_image[i - 1:i + 2, j - 1:j + 2]
        filtered_value = np.mean(neighborhood)
        filtered_image[i, j] = filtered_value
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image (Salt-and-Pepper)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image (Manual Box Filter 3x3)')
plt.axis('off')

plt.tight_layout()
plt.show()

```
ii) Using Weighted Averaging Filter
```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('fish.png', cv2.IMREAD_GRAYSCALE)

salt_prob = 0.05 
pepper_prob = 0.05  

noisy_image = np.copy(image)

num_salt = np.ceil(salt_prob * image.size)
coords_salt = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
noisy_image[tuple(coords_salt)] = 255

num_pepper = np.ceil(pepper_prob * image.size)
coords_pepper = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
noisy_image[tuple(coords_pepper)] = 0

kernel = np.array([[1, 2, 1],
                   [2, 4, 2],
                   [1, 2, 1]]) / 16.0 

image_height, image_width = noisy_image.shape
kernel_size = kernel.shape[0]  
pad = kernel_size // 2

padded_image = np.pad(noisy_image, pad, mode='constant', constant_values=0)

filtered_image = np.zeros_like(noisy_image)

for i in range(pad, image_height + pad):
    for j in range(pad, image_width + pad):
        roi = padded_image[i - pad:i + pad + 1, j - pad:j + pad + 1]
        filtered_value = np.sum(roi * kernel)
        filtered_image[i - pad, j - pad] = np.clip(filtered_value, 0, 255)
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image (Salt-and-Pepper)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image (Manual Weighted Avg)')
plt.axis('off')

plt.tight_layout()
plt.show()
```
iii)Using Median Filter
```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('fish.png', cv2.IMREAD_GRAYSCALE)

salt_prob = 0.05 
pepper_prob = 0.05 

noisy_image = np.copy(image)

num_salt = np.ceil(salt_prob * image.size)
coords_salt = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
noisy_image[tuple(coords_salt)] = 255

num_pepper = np.ceil(pepper_prob * image.size)
coords_pepper = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
noisy_image[tuple(coords_pepper)] = 0

filtered_image = np.zeros_like(noisy_image)  # Create an empty output image
height, width = noisy_image.shape

for i in range(1, height - 1):
    for j in range(1, width - 1):
        neighborhood = noisy_image[i - 1:i + 2, j - 1:j + 2]
        median_value = np.median(neighborhood)
        filtered_image[i, j] = median_value
        
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image (Salt-and-Pepper)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image (Manual Median Filter)')
plt.axis('off')

plt.tight_layout()
plt.show()
```

### 2. Sharpening Filters
i) Using Laplacian Linear Kernal
```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('fish.png', cv2.IMREAD_GRAYSCALE)

blurred_image = cv2.GaussianBlur(image, (3, 3), 0)

laplacian_kernel = np.array([[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]])

image_height, image_width = blurred_image.shape
kernel_height, kernel_width = laplacian_kernel.shape

pad_height = kernel_height // 2
pad_width = kernel_width // 2

padded_image = np.pad(blurred_image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
laplacian_image = np.zeros_like(blurred_image)

for i in range(image_height):
    for j in range(image_width):
        region = padded_image[i:i + kernel_height, j:j + kernel_width]
        laplacian_value = np.sum(region * laplacian_kernel)
        laplacian_image[i, j] = laplacian_value

laplacian_image = np.clip(laplacian_image, 0, 255).astype(np.uint8)
sharpened_image = cv2.add(image, laplacian_image)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(laplacian_image, cmap='gray')
plt.title('Laplacian Filtered Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sharpened_image, cmap='gray')
plt.title('Sharpened Image')
plt.axis('off')

plt.tight_layout()
plt.show()
```
ii) Using Laplacian Operator
1) Sobel Operator:
```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('fish.png', cv2.IMREAD_GRAYSCALE)

blurred_image = cv2.GaussianBlur(image, (3, 3), 0)

sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)

sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

sobel_magnitude = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX)
sobel_magnitude = np.uint8(sobel_magnitude)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.imshow(blurred_image, cmap='gray')
plt.title('Blurred Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(sobel_magnitude, cmap='gray')
plt.title('Sobel Operator')
plt.axis('off')

plt.tight_layout()
plt.show()
```
2) Roberts Filter
```
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('fish.png', cv2.IMREAD_GRAYSCALE)

blurred_image = cv2.GaussianBlur(image, (3, 3), 0)

roberts_x = np.array([[1, 0], [0, -1]])
roberts_y = np.array([[0, 1], [-1, 0]])

gradient_x = cv2.filter2D(blurred_image, cv2.CV_64F, roberts_x)
gradient_y = cv2.filter2D(blurred_image, cv2.CV_64F, roberts_y)

roberts_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

roberts_magnitude = cv2.normalize(roberts_magnitude, None, 0, 255, cv2.NORM_MINMAX)
roberts_magnitude = np.uint8(roberts_magnitude)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.imshow(blurred_image, cmap='gray')
plt.title('Blurred Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(roberts_magnitude, cmap='gray')
plt.title('Roberts Filter')
plt.axis('off')

plt.tight_layout()
plt.show()
```
3) Prewitt Operator:
```
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('fish.png', cv2.IMREAD_GRAYSCALE)

blurred_image = cv2.GaussianBlur(image, (3, 3), 0)

prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])

prewitt_y = np.array([[-1, -1, -1],
                      [ 0,  0,  0],
                      [ 1,  1,  1]])

gradient_x = cv2.filter2D(blurred_image, cv2.CV_64F, prewitt_x)
gradient_y = cv2.filter2D(blurred_image, cv2.CV_64F, prewitt_y)

prewitt_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

prewitt_magnitude = cv2.normalize(prewitt_magnitude, None, 0, 255, cv2.NORM_MINMAX)
prewitt_magnitude = np.uint8(prewitt_magnitude)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.imshow(blurred_image, cmap='gray')
plt.title('Blurred Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(prewitt_magnitude, cmap='gray')
plt.title('Prewitt Operator')
plt.axis('off')

plt.tight_layout()
plt.show()
```
4) Gradient Magnitude:
```
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('fish.png', cv2.IMREAD_GRAYSCALE)

blurred_image = cv2.GaussianBlur(image, (3, 3), 0)

grad_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)

gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
gradient_magnitude = np.uint8(gradient_magnitude)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.imshow(blurred_image, cmap='gray')
plt.title('Blurred Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(gradient_magnitude, cmap='gray')
plt.title('Gradient Magnitude')
plt.axis('off')

plt.tight_layout()
plt.show()
```

## OUTPUT:
### 1. Smoothing Filters
</br>

i) Using Averaging Filter
![image](https://github.com/user-attachments/assets/270ce43c-4794-4fdd-bccf-aba6a10f722d)


ii)Using Weighted Averaging Filter
![image](https://github.com/user-attachments/assets/d3905e19-b43a-4ab9-892a-d40041675276)


iii) Using Median Filter
![image](https://github.com/user-attachments/assets/ba94e149-6111-444d-8560-6913a7089c02)


### 2. Sharpening Filters
</br>

i) Using Laplacian Kernal
![image](https://github.com/user-attachments/assets/08b7af43-49fe-4929-8a7f-40a70ea1fe03)


ii) Using Laplacian Operator
![image](https://github.com/user-attachments/assets/5dfa52c4-f170-406b-93a1-6f4a61afb4a1)
![image](https://github.com/user-attachments/assets/fecddb60-2670-4458-8c26-42d36076e8c2)
![image](https://github.com/user-attachments/assets/9f872753-2a18-40de-9603-0ba22c0d6d41)
![image](https://github.com/user-attachments/assets/8f4536e2-68a1-4e09-882d-a2962ba99d9a)


## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
