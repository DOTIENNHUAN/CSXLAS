import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

#Đặt biến đường dẫn tới ảnh
path = r'N:\B2_CSXLAS\img\ronaldo.jpg'

# 1. Đọc ảnh đầu vào
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# 2. Phát hiện biên bằng Canny
edges = cv2.Canny(image, 100, 200)

# 3. Phân vùng ảnh bằng Thresholding (Otsu)
# Tính ngưỡng Otsu và phân vùng
_, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 4. Hiển thị kết quả
plt.figure(figsize=(12, 6))

# Ảnh gốc
plt.subplot(1, 3, 1)
plt.title("Ảnh gốc")
plt.imshow(image, cmap='gray')
plt.axis('off')

# Ảnh sau phát hiện biên
plt.subplot(1, 3, 2)
plt.title("Phát hiện biên (Canny)")
plt.imshow(edges, cmap='gray')
plt.axis('off')

# Ảnh sau phân vùng
plt.subplot(1, 3, 3)
plt.title("Phân vùng (Otsu)")
plt.imshow(thresholded, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# 5. Lưu ảnh vào thư mục "output_images"
output_dir = "output_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
cv2.imwrite(os.path.join(output_dir, 'original_image.jpg'), image)
cv2.imwrite(os.path.join(output_dir, 'edges_canny.jpg'), edges)
cv2.imwrite(os.path.join(output_dir, 'segmentation_otsu.jpg'), thresholded)

print(f"Đã lưu kết quả vào thư mục: {output_dir}")
