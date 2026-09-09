import os
import io
import cv2
import joblib
import random
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageChops, ImageEnhance, ImageFilter

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = "ud-biometrics/passport-dataset"

PARQUET_URL = (
    "https://huggingface.co/api/datasets/"
    "ud-biometrics/passport-dataset/"
    "parquet/default/train/0.parquet"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
GENUINE_DIR = os.path.join(DATA_DIR, "genuine")
TAMPERED_DIR = os.path.join(DATA_DIR, "tampered")

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")

MODEL_DIR = os.path.join(BASE_DIR, "models")

PARQUET_FILE = os.path.join(
    BASE_DIR,
    "passport_dataset.parquet"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "document_authenticity_model.pkl"
)


# Number of tampered versions generated per genuine image

TAMPERS_PER_IMAGE = 4

RANDOM_STATE = 42


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_directories():

    directories = [
        DATA_DIR,
        GENUINE_DIR,
        TAMPERED_DIR,
        OUTPUT_DIR,
        VIS_DIR,
        MODEL_DIR
    ]

    for directory in directories:

        os.makedirs(
            directory,
            exist_ok=True
        )


# ============================================================
# DOWNLOAD PARQUET DATASET
# ============================================================

def download_dataset():

    print("\n========================================")
    print("DOWNLOADING HUGGING FACE DATASET")
    print("========================================")

    if os.path.exists(PARQUET_FILE):

        print(
            "Parquet file already exists."
        )

        return

    response = requests.get(
        PARQUET_URL,
        timeout=120
    )

    response.raise_for_status()

    with open(
        PARQUET_FILE,
        "wb"
    ) as file:

        file.write(
            response.content
        )

    print(
        "Downloaded:",
        PARQUET_FILE
    )


# ============================================================
# READ PARQUET
# ============================================================

def read_dataset():

    print("\nReading Parquet file...")

    df = pd.read_parquet(
        PARQUET_FILE
    )

    print(
        "Rows:",
        len(df)
    )

    print(
        "Columns:",
        df.columns.tolist()
    )

    return df


# ============================================================
# EXTRACT IMAGE BYTES
# ============================================================

def extract_image_bytes(image_value):

    """
    Handles Hugging Face image columns.

    Depending on the Parquet conversion,
    image_value may be:

        dict
        bytes
        numpy object
    """

    # ------------------------------------------
    # Dictionary
    # ------------------------------------------

    if isinstance(
        image_value,
        dict
    ):

        image_bytes = image_value.get(
            "bytes"
        )

        if image_bytes is not None:

            return image_bytes

        image_path = image_value.get(
            "path"
        )

        if image_path and os.path.exists(
            image_path
        ):

            with open(
                image_path,
                "rb"
            ) as file:

                return file.read()

    # ------------------------------------------
    # Bytes
    # ------------------------------------------

    if isinstance(
        image_value,
        bytes
    ):

        return image_value

    # ------------------------------------------
    # Bytearray
    # ------------------------------------------

    if isinstance(
        image_value,
        bytearray
    ):

        return bytes(
            image_value
        )

    return None


# ============================================================
# SAVE GENUINE IMAGES
# ============================================================

def extract_genuine_images(df):

    print("\n========================================")
    print("EXTRACTING GENUINE IMAGES")
    print("========================================")

    if "image" not in df.columns:

        raise ValueError(
            "No 'image' column found."
        )

    saved = []

    for index, row in df.iterrows():

        image_bytes = extract_image_bytes(
            row["image"]
        )

        if image_bytes is None:

            print(
                f"Could not extract image {index}"
            )

            continue

        try:

            image = Image.open(
                io.BytesIO(image_bytes)
            )

            image = image.convert(
                "RGB"
            )

            filename = (
                f"genuine_{index:04d}.jpg"
            )

            filepath = os.path.join(
                GENUINE_DIR,
                filename
            )

            image.save(
                filepath,
                "JPEG",
                quality=95
            )

            saved.append(
                filepath
            )

            print(
                f"Saved: {filename}"
            )

        except Exception as error:

            print(
                f"Error processing image {index}:",
                error
            )

    print(
        f"\nGenuine images saved: {len(saved)}"
    )

    return saved


# ============================================================
# RANDOM RECTANGLE
# ============================================================

def random_rectangle(
    width,
    height
):

    # Rectangle size
    rect_width = int(
        width * random.uniform(
            0.08,
            0.25
        )
    )

    rect_height = int(
        height * random.uniform(
            0.08,
            0.25
        )
    )

    x1 = random.randint(
        0,
        max(0, width - rect_width)
    )

    y1 = random.randint(
        0,
        max(0, height - rect_height)
    )

    x2 = min(
        width,
        x1 + rect_width
    )

    y2 = min(
        height,
        y1 + rect_height
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# TAMPERING METHOD 1
# COPY / PASTE REGION
# ============================================================

def tamper_copy_paste(image):

    image = image.copy()

    width, height = image.size

    source_box = random_rectangle(
        width,
        height
    )

    target_box = random_rectangle(
        width,
        height
    )

    source = image.crop(
        source_box
    )

    target_width = (
        target_box[2] -
        target_box[0]
    )

    target_height = (
        target_box[3] -
        target_box[1]
    )

    source = source.resize(
        (
            target_width,
            target_height
        )
    )

    image.paste(
        source,
        (
            target_box[0],
            target_box[1]
        )
    )

    return image


# ============================================================
# TAMPERING METHOD 2
# BLUR REGION
# ============================================================

def tamper_blur(image):

    image = image.copy()

    width, height = image.size

    box = random_rectangle(
        width,
        height
    )

    region = image.crop(
        box
    )

    region = region.filter(
        ImageFilter.GaussianBlur(
            radius=4
        )
    )

    image.paste(
        region,
        (
            box[0],
            box[1]
        )
    )

    return image


# ============================================================
# TAMPERING METHOD 3
# COLOR MANIPULATION
# ============================================================

def tamper_color(image):

    image = image.copy()

    width, height = image.size

    box = random_rectangle(
        width,
        height
    )

    region = image.crop(
        box
    )

    # Change brightness
    region = ImageEnhance.Brightness(
        region
    ).enhance(
        random.uniform(
            0.55,
            1.5
        )
    )

    image.paste(
        region,
        (
            box[0],
            box[1]
        )
    )

    return image


# ============================================================
# TAMPERING METHOD 4
# JPEG RECOMPRESSION
# ============================================================

def tamper_recompression(image):

    buffer = io.BytesIO()

    quality = random.choice(
        [
            20,
            30,
            40,
            50
        ]
    )

    image.save(
        buffer,
        format="JPEG",
        quality=quality
    )

    buffer.seek(0)

    compressed = Image.open(
        buffer
    ).convert(
        "RGB"
    )

    return compressed


# ============================================================
# CREATE TAMPERED DATASET
# ============================================================

def create_tampered_dataset(
    genuine_images
):

    print("\n========================================")
    print("CREATING TAMPERED IMAGES")
    print("========================================")

    methods = [
        tamper_copy_paste,
        tamper_blur,
        tamper_color,
        tamper_recompression
    ]

    created = []

    for image_number, filepath in enumerate(
        genuine_images
    ):

        image = Image.open(
            filepath
        ).convert(
            "RGB"
        )

        for tamper_number in range(
            TAMPERS_PER_IMAGE
        ):

            method = methods[
                tamper_number %
                len(methods)
            ]

            try:

                tampered = method(
                    image
                )

                filename = (
                    f"tampered_"
                    f"{image_number:04d}_"
                    f"{tamper_number}.jpg"
                )

                output_path = os.path.join(
                    TAMPERED_DIR,
                    filename
                )

                tampered.save(
                    output_path,
                    "JPEG",
                    quality=90
                )

                created.append(
                    output_path
                )

            except Exception as error:

                print(
                    "Tampering error:",
                    error
                )

    print(
        f"Tampered images created: {len(created)}"
    )

    return created


# ============================================================
# ELA
# ============================================================

def calculate_ela(
    image_path,
    quality=90
):

    original = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    buffer = io.BytesIO()

    original.save(
        buffer,
        format="JPEG",
        quality=quality
    )

    buffer.seek(0)

    compressed = Image.open(
        buffer
    ).convert(
        "RGB"
    )

    difference = ImageChops.difference(
        original,
        compressed
    )

    extrema = difference.getextrema()

    max_difference = max(
        channel[1]
        for channel in extrema
    )

    if max_difference == 0:

        max_difference = 1

    scale = 255.0 / max_difference

    ela_image = ImageEnhance.Brightness(
        difference
    ).enhance(
        scale
    )

    ela_array = np.array(
        ela_image
    ).astype(
        np.float32
    )

    gray_ela = cv2.cvtColor(
        ela_array.astype(
            np.uint8
        ),
        cv2.COLOR_RGB2GRAY
    )

    return (
        ela_array,
        gray_ela
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    image_path
):

    image = cv2.imread(
        image_path
    )

    if image is None:

        raise ValueError(
            f"Could not read {image_path}"
        )

    # ------------------------------------------
    # Resize
    # ------------------------------------------

    image = cv2.resize(
        image,
        (
            800,
            600
        )
    )

    # ------------------------------------------
    # Gray
    # ------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # ------------------------------------------
    # ELA
    # ------------------------------------------

    ela, ela_gray = calculate_ela(
        image_path
    )

    ela_mean = float(
        np.mean(ela)
    )

    ela_std = float(
        np.std(ela)
    )

    ela_max = float(
        np.max(ela)
    )

    # Percentage of high ELA pixels
    ela_high_pixels = float(
        np.mean(
            ela_gray > 40
        )
    )

    # ------------------------------------------
    # Texture
    # ------------------------------------------

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    texture_score = float(
        laplacian.var()
    )

    # ------------------------------------------
    # Edges
    # ------------------------------------------

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_density = float(
        np.mean(
            edges > 0
        )
    )

    # ------------------------------------------
    # Color
    # ------------------------------------------

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    hue_std = float(
        np.std(
            hsv[:, :, 0]
        )
    )

    saturation_std = float(
        np.std(
            hsv[:, :, 1]
        )
    )

    brightness_std = float(
        np.std(
            hsv[:, :, 2]
        )
    )

    # ------------------------------------------
    # Blur
    # ------------------------------------------

    blur_score = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    # ------------------------------------------
    # Noise estimation
    # ------------------------------------------

    noise = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    noise_score = float(
        np.std(noise)
    )

    # ------------------------------------------
    # Image dimensions
    # ------------------------------------------

    height, width = gray.shape

    aspect_ratio = float(
        width / height
    )

    # ------------------------------------------
    # Final feature vector
    # ------------------------------------------

    features = [

        ela_mean,
        ela_std,
        ela_max,
        ela_high_pixels,

        texture_score,
        edge_density,

        hue_std,
        saturation_std,
        brightness_std,

        blur_score,
        noise_score,

        width,
        height,
        aspect_ratio
    ]

    return np.array(
        features,
        dtype=np.float32
    )


# ============================================================
# BUILD ML DATASET
# ============================================================

def build_feature_dataset():

    print("\n========================================")
    print("EXTRACTING FEATURES")
    print("========================================")

    rows = []

    # ------------------------------------------
    # Genuine
    # ------------------------------------------

    genuine_files = [
        os.path.join(
            GENUINE_DIR,
            file
        )

        for file in os.listdir(
            GENUINE_DIR
        )

        if file.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png"
            )
        )
    ]

    for filepath in genuine_files:

        try:

            features = extract_features(
                filepath
            )

            source_id = os.path.basename(
                filepath
            ).split("_")[1].split(".")[0]

            rows.append(
                {
                    "filepath": filepath,
                    "label": 0,
                    "source_id": source_id,
                    **{
                        f"feature_{i}": value
                        for i, value in enumerate(
                            features
                        )
                    }
                }
            )

        except Exception as error:

            print(
                "Feature error:",
                filepath,
                error
            )

    # ------------------------------------------
    # Tampered
    # ------------------------------------------

    tampered_files = [
        os.path.join(
            TAMPERED_DIR,
            file
        )

        for file in os.listdir(
            TAMPERED_DIR
        )

        if file.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png"
            )
        )
    ]

    for filepath in tampered_files:

        try:

            features = extract_features(
                filepath
            )

            filename = os.path.basename(
                filepath
            )

            # tampered_0000_0.jpg
            parts = filename.split("_")

            source_id = parts[1]

            rows.append(
                {
                    "filepath": filepath,
                    "label": 1,
                    "source_id": source_id,
                    **{
                        f"feature_{i}": value
                        for i, value in enumerate(
                            features
                        )
                    }
                }
            )

        except Exception as error:

            print(
                "Feature error:",
                filepath,
                error
            )

    dataset = pd.DataFrame(
        rows
    )

    csv_path = os.path.join(
        OUTPUT_DIR,
        "features.csv"
    )

    dataset.to_csv(
        csv_path,
        index=False
    )

    print(
        "\nFeature dataset saved:"
    )

    print(csv_path)

    print(
        "\nClass distribution:"
    )

    print(
        dataset["label"].value_counts()
    )

    return dataset


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

def train_model(dataset):

    print("\n========================================")
    print("TRAINING RANDOM FOREST")
    print("========================================")

    feature_columns = [
        column
        for column in dataset.columns
        if column.startswith(
            "feature_"
        )
    ]

    X = dataset[
        feature_columns
    ]

    y = dataset[
        "label"
    ]

    groups = dataset[
        "source_id"
    ]

    # ------------------------------------------
    # Group split
    #
    # Important:
    # Genuine image and its tampered copies
    # stay in the same train/test group.
    # ------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=RANDOM_STATE
    )

    train_indices, test_indices = next(
        splitter.split(
            X,
            y,
            groups
        )
    )

    X_train = X.iloc[
        train_indices
    ]

    X_test = X.iloc[
        test_indices
    ]

    y_train = y.iloc[
        train_indices
    ]

    y_test = y.iloc[
        test_indices
    ]

    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Testing samples:",
        len(X_test)
    )

    # ------------------------------------------
    # Random Forest
    # ------------------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    # ------------------------------------------
    # Prediction
    # ------------------------------------------

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Genuine",
                "Tampered"
            ],
            zero_division=0
        )
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    # ------------------------------------------
    # Save model
    # ------------------------------------------

    model_data = {

        "model": model,

        "features": feature_columns,

        "description": (
            "Document authenticity classifier"
        )
    }

    joblib.dump(
        model_data,
        MODEL_FILE
    )

    print(
        "\nModel saved:"
    )

    print(
        MODEL_FILE
    )

    # ------------------------------------------
    # Feature importance
    # ------------------------------------------

    importance = model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importance
        }
    ).sort_values(
        "importance",
        ascending=False
    )

    print(
        "\nFeature importance:"
    )

    print(
        importance_df
    )

    return model


# ============================================================
# CREATE ELA VISUALIZATION
# ============================================================

def create_visualization(
    image_path
):

    print(
        "\nCreating visualization..."
    )

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    ela, ela_gray = calculate_ela(
        image_path
    )

    cv_image = cv2.imread(
        image_path
    )

    cv_image = cv2.resize(
        cv_image,
        (
            800,
            600
        )
    )

    gray = cv2.cvtColor(
        cv_image,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    plt.figure(
        figsize=(15, 5)
    )

    # Original
    plt.subplot(
        1,
        3,
        1
    )

    plt.imshow(
        image
    )

    plt.title(
        "Original Document"
    )

    plt.axis(
        "off"
    )

    # ELA
    plt.subplot(
        1,
        3,
        2
    )

    plt.imshow(
        ela.astype(
            np.uint8
        )
    )

    plt.title(
        "ELA Analysis"
    )

    plt.axis(
        "off"
    )

    # Edges
    plt.subplot(
        1,
        3,
        3
    )

    plt.imshow(
        edges,
        cmap="gray"
    )

    plt.title(
        "Edge Analysis"
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    filename = os.path.join(
        VIS_DIR,
        "document_analysis.png"
    )

    plt.savefig(
        filename,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    print(
        "Visualization saved:"
    )

    print(
        filename
    )


# ============================================================
# PREDICT NEW DOCUMENT
# ============================================================

def predict_document(
    image_path
):

    print("\n========================================")
    print("DOCUMENT AUTHENTICITY PREDICTION")
    print("========================================")

    if not os.path.exists(
        MODEL_FILE
    ):

        raise FileNotFoundError(
            "Model not found. "
            "Train the model first."
        )

    # Load model
    model_data = joblib.load(
        MODEL_FILE
    )

    model = model_data[
        "model"
    ]

    feature_columns = model_data[
        "features"
    ]

    # Extract features
    features = extract_features(
        image_path
    )

    X = pd.DataFrame(
        [
            features
        ],
        columns=feature_columns
    )

    # Probability
    probabilities = model.predict_proba(
        X
    )[0]

    genuine_probability = (
        probabilities[0] * 100
    )

    tampered_probability = (
        probabilities[1] * 100
    )

    # ------------------------------------------
    # Risk score
    # ------------------------------------------

    risk_score = tampered_probability

    # ------------------------------------------
    # Decision
    # ------------------------------------------

    if risk_score < 30:

        decision = "LIKELY GENUINE"

    elif risk_score < 70:

        decision = "REVIEW REQUIRED"

    else:

        decision = "SUSPICIOUS / POSSIBLE TAMPERING"

    print(
        "\nGenuine Probability:",
        f"{genuine_probability:.2f}%"
    )

    print(
        "Tampering Probability:",
        f"{tampered_probability:.2f}%"
    )

    print(
        "Risk Score:",
        f"{risk_score:.2f}%"
    )

    print(
        "Decision:",
        decision
    )

    # ------------------------------------------
    # ELA visualization
    # ------------------------------------------

    create_visualization(
        image_path
    )

    return {

        "genuine_probability":
            genuine_probability,

        "tampered_probability":
            tampered_probability,

        "risk_score":
            risk_score,

        "decision":
            decision
    }


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    random.seed(
        RANDOM_STATE
    )

    np.random.seed(
        RANDOM_STATE
    )

    create_directories()

    # ------------------------------------------
    # STEP 1
    # Download dataset
    # ------------------------------------------

    download_dataset()

    # ------------------------------------------
    # STEP 2
    # Read dataset
    # ------------------------------------------

    df = read_dataset()

    # ------------------------------------------
    # STEP 3
    # Extract genuine images
    # ------------------------------------------

    genuine_images = extract_genuine_images(
        df
    )

    if len(genuine_images) == 0:

        raise RuntimeError(
            "No images were extracted."
        )

    # ------------------------------------------
    # STEP 4
    # Create tampered images
    # ------------------------------------------

    tampered_images = create_tampered_dataset(
        genuine_images
    )

    # ------------------------------------------
    # STEP 5
    # Feature extraction
    # ------------------------------------------

    feature_dataset = build_feature_dataset()

    # ------------------------------------------
    # STEP 6
    # Train model
    # ------------------------------------------

    train_model(
        feature_dataset
    )

    print("\n========================================")
    print("TRAINING COMPLETE")
    print("========================================")

    print(
        "\nModel:",
        MODEL_FILE
    )

    print(
        "\nTo test another document:"
    )

    print(
        "Use predict_document('your_image.jpg')"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
    
from document_authenticity import predict_document

result = predict_document(
    "Screenshot 2026-09-08 234319.png"
)

print(result)
