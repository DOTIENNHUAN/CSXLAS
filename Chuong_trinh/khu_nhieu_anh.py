import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

#Đặt biến đường dẫn tới ảnh
path = r'N:\B2_CSXLAS\img\ronaldo.jpg'

# 1. Đọc ảnh đầu vào
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# 2. Thêm nhiễu muối và tiêu vào ảnh
def add_salt_and_pepper_noise(image, prob):
    noisy_image = image.copy()
    total_pixels = image.size
    num_salt = int(prob * total_pixels / 2)
    num_pepper = int(prob * total_pixels / 2)
    
    # Tạo nhiễu muối (điểm trắng)
    coords_salt = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[coords_salt[0], coords_salt[1]] = 255
    
    # Tạo nhiễu tiêu (điểm đen)
    coords_pepper = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[coords_pepper[0], coords_pepper[1]] = 0
    
    return noisy_image

noisy_image = add_salt_and_pepper_noise(image, prob=0.05)

# 3. Áp dụng bộ lọc trung vị
denoised_image = cv2.medianBlur(noisy_image, ksize=3)

# 4. Hiển thị ảnh
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title("Ảnh gốc")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Ảnh có nhiễu")
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Ảnh đã khử nhiễu")
plt.imshow(denoised_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()


# 5. Lưu ảnh vào thư mục "output_images"
output_dir = "output_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cv2.imwrite(os.path.join(output_dir, 'original_image.jpg'), image)
cv2.imwrite(os.path.join(output_dir, 'noisy_image.jpg'), noisy_image)
cv2.imwrite(os.path.join(output_dir, 'denoised_image.jpg'), denoised_image)

print(f"Đã lưu ảnh vào thư mục: {output_dir}")
