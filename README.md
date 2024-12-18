# object_detect_with_YOLO

you can use google colabs to run the code

# install YOLO
`
!pip install ultralytics opencv-python-headless matplotlib --quiet
`
After installing run the code
# Code
`python

# 2. Import library
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from google.colab import files

# 3. Unggah gambar
print("Silakan unggah gambar yang akan digunakan:")
uploaded = files.upload()

# Ambil nama file gambar yang diunggah
image_path = list(uploaded.keys())[0]

# 4. Load model YOLO
model = YOLO("yolov8n.pt")  # YOLOv5s pretrained model

# 5. Fungsi untuk deteksi objek
def detect_objects(image_path):
    # Baca gambar
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Deteksi objek
    results = model(image_path)

    # Plot hasil deteksi
    annotated_img = results[0].plot()  # Gambar dengan bounding box
    plt.imshow(annotated_img)
    plt.axis('off')
    plt.show()

    # Print hasil deteksi ke terminal
    print("Hasil Deteksi:")
    for result in results[0].boxes:
        label = model.names[int(result.cls)]  # Nama label
        confidence = result.conf.item()  # Confidence score
        coordinates = result.xyxy.cpu().numpy()  # Koordinat bounding box
        print(f"Label: {label}, Confidence: {confidence:.2f}, Coordinates: {coordinates}")

# 6. Jalankan deteksi objek
detect_objects(image_path)

`
