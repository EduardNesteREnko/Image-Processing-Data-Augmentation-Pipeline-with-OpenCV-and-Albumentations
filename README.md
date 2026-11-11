# Image Processing & Data Augmentation Pipeline with OpenCV and Albumentations

This repository contains a Python script for image loading, manipulation, basic transformation, and advanced data augmentation using **OpenCV**, **Matplotlib**, and the **Albumentations** library.

## 📋 Overview of the Pipeline

1. **Image Loading & Color Conversion (`load_image`)**:
   * Reads an image from disk using OpenCV (`cv2.imread`).
   * Converts the default BGR color space to RGB to ensure correct visualization via Matplotlib.
   * Handles missing file exceptions gracefully.

2. **Basic Image Processing**:
   * **Resizing**: Reshapes images to a uniform $400 \times 400$ resolution using `cv2.resize`.
   * **Horizontal Flipping**: Mirrors the image horizontally using `cv2.flip`.

3. **Advanced Data Augmentation (`Albumentations`)**:
   * Applies a customizable augmentation pipeline (`A.Compose`) featuring:
     * **Random Crop**: Crops a $300 \times 300$ patch ($p=1.0$).
     * **Random Brightness & Contrast**: Adjusts lighting dynamically ($p=0.8$).
     * **Gaussian Noise**: Adds synthetic sensor noise ($p=0.8$).
     * **Rotation**: Rotates images within a $\pm 25^{\circ}$ limit ($p=0.5$).

4. **Batch Processing & Visualization**:
   * Iterates through a specified list of image paths.
   * Displays side-by-side plots for original, resized, and flipped versions, followed by multi-sample augmented variations.

---

## 🛠️ Code Implementation

```python
import cv2
import matplotlib.pyplot as plt
import albumentations as A

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(
            "File not found. Please check the path and try again.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def display_image(images, titles=None, figsize=(15, 5)):
    n = len(images)
    plt.figure(figsize=figsize)
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i])
        if titles and i < len(titles):
            plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def augment_image(image, transform_pipeline, num_samples=3):
    augmented_images = []
    for _ in range(num_samples):
        augmented = transform_pipeline(image=image)
        augmented_images.append(augmented['image'])
    return augmented_images

image_paths = ['Dwayne.jpg', 'gordon.jpeg', 'JasonStatham.jpeg', 'Vin_Diesel.jpeg']

transform = A.Compose([
    A.RandomCrop(width=300, height=300, p=1.0), 
    A.RandomBrightnessContrast(p=0.8),  
    A.GaussNoise(p=0.8),  
    A.Rotate(limit=25, p=0.5)   
])

for i, path in enumerate(image_paths):
    print(f"\nProcessing [{i + 1}/4]: {path}")

    try:
        original_image = load_image(path)

        h, w = original_image.shape[:2]
        resized_img = cv2.resize(original_image, (400, 400))
        flipped_img = cv2.flip(original_image, 1)  

        display_image(
            [original_image, resized_img, flipped_img],
            titles=[f'Origin {i + 1}', 'Resize (400x400)', 'Flip'],
            figsize=(12, 4)
        )

        augmented_results = augment_image(original_image, transform, num_samples=3)

        display_image(
            [original_image] + augmented_results,
            titles=[f'Origin {i + 1}', 'Augmentation1', 'Augmentation2', 'Augmentation3'],
            figsize=(15, 4)
        )

    except FileNotFoundError as e:
        print(f"⚠️ missing: {e}")
```

---

## 📊 Features & Benefits

* **Robust Exception Handling**: Automatically catches missing image files without interrupting the entire batch loop.
* **Flexible Transformations**: Easily extendable pipeline via `Albumentations` for computer vision model training data augmentation.