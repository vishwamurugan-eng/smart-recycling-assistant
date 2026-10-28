"""
Supervised Waste Image Classifier Training Script (TensorFlow + scikit-learn).
Trains a 4-class classifier (glass, metal, paper, plastic), evaluates it with
scikit-learn metrics, logs the dataset metadata to MySQL, and saves artifacts.
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from db_manager import sync_dataset_to_mysql

CLASSES = ["glass", "metal", "paper", "plastic"]
IMG_SIZE = (160, 160)
BATCH_SIZE = 16
MODEL_DIR = "artifacts"


def generate_synthetic_dataset(dataset_dir="dataset", images_per_class=40):
    """
    Generates distinct synthetic waste images per category so you can test the
    end-to-end pipeline immediately before downloading a large Kaggle dataset.
    """
    rng = np.random.default_rng(42)
    palettes = {
        "glass": (180, 230, 235),    # Translucent cyan/teal bottle tones
        "metal": (165, 175, 185),    # Metallic silver/gray can tones
        "paper": (235, 220, 190),    # Cardboard/kraft paper beige tones
        "plastic": (60, 140, 230),   # Bright plastic bottle blue/orange tones
    }

    for label in CLASSES:
        folder = os.path.join(dataset_dir, label)
        os.makedirs(folder, exist_ok=True)
        base_r, base_g, base_b = palettes[label]

        for i in range(images_per_class):
            noise = rng.integers(-25, 25, size=(IMG_SIZE[1], IMG_SIZE[0], 3))
            bg = np.clip(
                np.array([base_r, base_g, base_b]) + noise, 0, 255
            ).astype(np.uint8)
            img = Image.fromarray(bg)
            draw = ImageDraw.Draw(img)

            # Draw distinct geometric silhouettes per material class
            cx, cy = IMG_SIZE[0] // 2 + int(rng.integers(-15, 15)), IMG_SIZE[1] // 2
            if label == "glass":
                # Bottle neck + body silhouette
                draw.rectangle([cx - 12, 25, cx + 12, 65], fill=(210, 245, 250), outline=(90, 150, 160), width=2)
                draw.rectangle([cx - 28, 65, cx + 28, 135], fill=(200, 240, 245), outline=(90, 150, 160), width=2)
            elif label == "metal":
                # Cylindrical aluminum can with metallic ridges
                draw.ellipse([cx - 30, 35, cx + 30, 55], fill=(210, 215, 220), outline=(90, 95, 100), width=2)
                draw.rectangle([cx - 30, 45, cx + 30, 125], fill=(190, 195, 205), outline=(90, 95, 100), width=2)
                draw.ellipse([cx - 30, 115, cx + 30, 135], fill=(175, 180, 190), outline=(90, 95, 100), width=2)
            elif label == "paper":
                # Rectangular sheet/cardboard with folded corner and lines
                draw.rectangle([cx - 38, 30, cx + 38, 130], fill=(245, 238, 215), outline=(160, 140, 110), width=2)
                for y_line in range(48, 120, 16):
                    draw.line([cx - 26, y_line, cx + 26, y_line], fill=(180, 165, 135), width=2)
            elif label == "plastic":
                # Tapered plastic container with bright cap
                draw.rectangle([cx - 10, 20, cx + 10, 34], fill=(240, 80, 50))
                draw.polygon(
                    [(cx - 16, 34), (cx + 16, 34), (cx + 34, 135), (cx - 34, 135)],
                    fill=(90, 170, 245),
                    outline=(40, 90, 170),
                )

            img.save(os.path.join(folder, f"{label}_{i+1:03d}.jpg"), quality=92)

    print(f"Created synthetic dataset in '{dataset_dir}/' ({images_per_class} images per class).")


def build_model(num_classes=4):
    """
    Builds a transfer-learning image classifier using MobileNetV2 with a custom
    classification head. Falls back to a lightweight custom CNN if offline.
    """
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.08)(x)

    try:
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
            include_top=False,
            weights="imagenet",
        )
        base_model.trainable = False
        x = base_model(preprocess, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.25)(x)
    except Exception:
        print("Using lightweight custom CNN architecture...")
        x = tf.keras.layers.Rescaling(1.0 / 255)(x)
        x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="smart_recycling_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_evaluate(dataset_dir="dataset", epochs=10):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Sync dataset filepaths with MySQL (non-blocking if MySQL isn't running yet)
    ok, db_msg = sync_dataset_to_mysql(dataset_dir)
    print(f"[Database Sync] {db_msg}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    class_names = train_ds.class_names
    print(f"Detected Classes: {class_names}")

    model = build_model(num_classes=len(class_names))
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    # Save trained TensorFlow model and class names
    model_path = os.path.join(MODEL_DIR, "recycling_cnn_model.keras")
    model.save(model_path)
    with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    # Evaluate on validation set using scikit-learn
    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    acc = accuracy_score(y_true, y_pred)
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    print("\n--- scikit-learn Classification Report ---")
    print(report_text)

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump({"accuracy": acc, "report": report_dict}, f, indent=2)

    # Plot & save scikit-learn Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Greens", ax=ax, colorbar=False)
    ax.set_title(f"Waste Classification Confusion Matrix (Acc: {acc*100:.1f}%)")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # Plot & save Training & Validation Accuracy/Loss Curves
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history.history["accuracy"], label="Train Acc", marker="o")
    ax1.plot(history.history["val_accuracy"], label="Val Acc", marker="s")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history.history["loss"], label="Train Loss", marker="o")
    ax2.plot(history.history["val_loss"], label="Val Loss", marker="s")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "training_curves.png"), dpi=150)
    plt.close(fig2)

    print(f"Saved model and evaluation plots to '{MODEL_DIR}/'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Smart Recycling Image Classifier")
    parser.add_argument(
        "--generate-sample-data",
        action="store_true",
        help="Generate synthetic sample images in dataset/ before training",
    )
    parser.add_argument("--dataset-dir", default="dataset", help="Path to dataset folder")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    args = parser.parse_args()

    if args.generate_sample_data or not os.path.exists(args.dataset_dir):
        generate_synthetic_dataset(args.dataset_dir)

    train_and_evaluate(args.dataset_dir, epochs=args.epochs)