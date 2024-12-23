import os
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import jaccard_score
from scipy.stats import norm
from PIL import Image
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
import cv2
from flask import Flask, render_template
from matplotlib.colors import ListedColormap
import threading

# Flask application setup
app = Flask(__name__)

# Create necessary directories if they don't exist
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

def load_images():
    file_paths = filedialog.askopenfilenames(title="Select CT Images", filetypes=[("BMP Files", "*.bmp")])
    if len(file_paths) != 4:
        messagebox.showerror("Error", "Please select exactly 4 CT images.")
        return None

    mask_paths = filedialog.askopenfilenames(title="Select Ground Truth Masks", filetypes=[("BMP Files", "*.bmp")])
    if len(mask_paths) != 4:
        messagebox.showerror("Error", "Please select exactly 4 ground truth masks.")
        return None

    # Load images and convert them to NumPy arrays
    ct_images = [np.array(Image.open(path)) for path in file_paths]
    masks = [np.array(Image.open(path)) for path in mask_paths]
    return np.array(ct_images), np.array(masks)

def bayesian_segmentation(ct_image, mask):
    mask = (mask > 140).astype(int)
    fg_pixels = ct_image[mask == 1]
    bg_pixels = ct_image[mask == 0]

    if len(fg_pixels) == 0 or len(bg_pixels) == 0:
        raise ValueError("Foreground or background pixel array is empty. Check your mask.")

    fg_mean, fg_std = np.mean(fg_pixels), np.std(fg_pixels)
    bg_mean, bg_std = np.mean(bg_pixels), np.std(bg_pixels)

    epsilon = 1e-8
    fg_std = max(fg_std, epsilon)
    bg_std = max(bg_std, epsilon)

    fg_prob = norm.pdf(ct_image, fg_mean, fg_std)
    bg_prob = norm.pdf(ct_image, bg_mean, bg_std)

    return (fg_prob > bg_prob).astype(int)

def calculate_dsc(ground_truth, prediction):
    intersection = np.sum(ground_truth * prediction)
    return (2.0 * intersection) / (np.sum(ground_truth) + np.sum(prediction))

def create_3d_plot(img):
    height, width = img.shape
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)
    Z = gaussian_filter(img, sigma=2)

    fig = go.Figure(data=go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale='gray',
        opacity=0.9,
    ))
    
    # Save the figure as an HTML file
    fig.write_html("./static/test_3d_projection.html")
    
    # Optionally, save as a static image (e.g., PNG)
    fig.write_image("static/test_3d_projection.png")  # Requires 'kaleido' package

def start_segmentation():
    data = load_images()
    if data is None:
        return 

    ct_images, masks = data
    kf = KFold(n_splits=2, shuffle=True, random_state=42)

    results = []
    for train_index, test_index in kf.split(ct_images):
        train_images, test_image = ct_images[train_index], ct_images[test_index][0]
        train_masks, test_mask = masks[train_index], masks[test_index][0]

        predictions = []
        for img, mask in zip(train_images, train_masks):
            predictions.append(bayesian_segmentation(img, mask))

        prediction = bayesian_segmentation(test_image, test_mask)
        dsc = calculate_dsc(test_mask, prediction)
        results.append(dsc)

        # Create 3D plot for the test image
        create_3d_plot(test_image)

        # Save the CT image, ground truth mask, and prediction
        plt.imsave("./static/test_image.png", test_image, cmap="gray")
        plt.imsave("./static/ground_truth_mask.png", test_mask, cmap="gray")
        plt.imsave("./static/prediction.png", prediction, cmap="gray")

    avg_dsc = np.mean(results)
    messagebox.showinfo("Results", f"Average DSC: {avg_dsc:.2f}")

    # Display last segmentation
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(test_image, cmap="gray")
    ax[0].set_title("CT Image")

    ax[1].imshow(test_mask, cmap="gray")
    ax[1].set_title("Ground Truth")

    ax[2].imshow(prediction, cmap="gray")
    ax[2].set_title("Prediction")

    plt.tight_layout()
    plt.show()


# Create the HTML template
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Plots</title>
</head>
<body>
    <style>
        .image-container {
            display: flex;
            justify-content: space-around;
            align-items: center;
        }
    </style>
    <h1>CT Lung Segmantation</h1>
    <div class="image-container">
        <div>
            <h1>CT Image</h1>
            <img src="./static/test_image.png" alt="CT Image">
        </div>
        <div>
            <h1>Ground Truth Mask</h1>
            <img src="./static/ground_truth_mask.png" alt="Ground Truth Mask">
        </div>
        <div>
            <h1>Segmanted Img</h1>
            <img src="./static/prediction.png" alt="Prediction">
        </div>
    </div>
    <h1>3D Plot</h1>
    <iframe src="./static/test_3d_projection.html" width="100%" height="600px"></iframe>
</body>
</html>
"""

# Save the HTML template
with open('templates/index.html', 'w') as f:
    f.write(html_content)

print("index.html created in templates directory.")






# GUI application
def main():
    root = tk.Tk()
    root.title("Lung Segmentation")

    load_button = tk.Button(root, text="Start Segmentation", command=start_segmentation)
    load_button.pack(pady=20)

    root.mainloop()

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    # Start the Flask app in a separate thread without the reloader
    flask_thread = threading.Thread(target=app.run, kwargs={'debug': True, 'use_reloader': False})
    flask_thread.start()

    # Start the Tkinter GUI
    main()



