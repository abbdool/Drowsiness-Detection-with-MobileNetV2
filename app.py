
import os
import cv2
import numpy as np
import tensorflow as tf
import dlib
import base64
import urllib.request
import bz2
import shutil
import time
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Limit request size to 8 MB
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

MODEL_PATH = "best_model_fold_5.keras"
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# DOWNLOAD DLIB LANDMARK MODEL

def download_landmark():
    if os.path.exists(PREDICTOR_PATH) and os.path.getsize(PREDICTOR_PATH) > 0:
        print("Landmark model already exists.")
        return

    print("Downloading shape_predictor_68_face_landmarks.dat ...")

    url = (
        "https://github.com/davisking/dlib-models/"
        "raw/master/shape_predictor_68_face_landmarks.dat.bz2"
    )

    compressed_file = PREDICTOR_PATH + ".bz2"

    try:
        urllib.request.urlretrieve(url, compressed_file)

        print("Extracting landmark model...")

        with bz2.BZ2File(compressed_file, "rb") as source:
            with open(PREDICTOR_PATH, "wb") as target:
                shutil.copyfileobj(source, target)

        os.remove(compressed_file)
        print("Landmark model downloaded successfully.")

    except Exception:
        app.logger.exception("Failed to download landmark model")
        raise


download_landmark()

# LOAD DLIB AND TENSORFLOW


print("Loading Dlib and TensorFlow models...")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
model_hybrid = tf.keras.models.load_model(MODEL_PATH)

print("Dlib and TensorFlow models loaded successfully.")

# CLASS AND LANDMARK CONFIG

CLASSES = ["Closed", "Open", "no_yawn", "yawn"]

LEFT_EYE = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]
MOUTH = [
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    58, 59, 60, 61, 62, 63, 64, 65, 66, 67
]

FRAME_WINDOW = 3
ear_history = []
mar_history = []
closed_frames = 0

ALARM_THRESHOLD = 2
EAR_THRESHOLD = 0.17
MAR_THRESHOLD = 0.70

# EAR CALCULATION

def calculate_ear(eye_points, landmarks):
    v1 = np.linalg.norm(
        np.array(landmarks[eye_points[1]]) -
        np.array(landmarks[eye_points[5]])
    )

    v2 = np.linalg.norm(
        np.array(landmarks[eye_points[2]]) -
        np.array(landmarks[eye_points[4]])
    )

    h = np.linalg.norm(
        np.array(landmarks[eye_points[0]]) -
        np.array(landmarks[eye_points[3]])
    )

    return (v1 + v2) / (2.0 * h) if h != 0 else 0.0


# MAR CALCULATION

def calculate_mar(mouth_points, landmarks):
    v1 = np.linalg.norm(
        np.array(landmarks[51]) - np.array(landmarks[59])
    )

    v2 = np.linalg.norm(
        np.array(landmarks[52]) - np.array(landmarks[58])
    )

    v3 = np.linalg.norm(
        np.array(landmarks[53]) - np.array(landmarks[57])
    )

    h = np.linalg.norm(
        np.array(landmarks[48]) - np.array(landmarks[54])
    )

    return (v1 + v2 + v3) / (3.0 * h) if h != 0 else 0.0


# HEAD POSE

def get_head_pose(landmarks, img_size):
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip (30)
        (0.0, -330.0, -65.0),        # Chin (8)
        (-225.0, 170.0, -135.0),     # Left eye corner (36)
        (225.0, 170.0, -135.0),      # Right eye corner (45)
        (-150.0, -150.0, -125.0),    # Left mouth corner (48)
        (150.0, -150.0, -125.0)      # Right mouth corner (54)
    ], dtype="double")

    image_points = np.array([
        landmarks[30],
        landmarks[8],
        landmarks[36],
        landmarks[45],
        landmarks[48],
        landmarks[54]
    ], dtype="double")

    height, width = img_size
    focal_length = width
    center = (width / 2, height / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    projection_matrix = np.hstack((
        rotation_matrix,
        translation_vector
    ))

    euler_angles = cv2.decomposeProjectionMatrix(
        projection_matrix
    )[6]

    pitch = float(euler_angles[0][0])
    yaw = float(euler_angles[1][0])
    roll = float(euler_angles[2][0])

    return pitch, yaw, roll


# FEATURE EXTRACTION
def extract_features(img):
    if img is None:
        raise ValueError("Invalid image")

    # Reduce the maximum image dimension to 640 pixels
    height, width = img.shape[:2]
    max_dimension = 640

    scale = min(
        max_dimension / max(height, width),
        1.0
    )

    if scale < 1.0:
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))

        img = cv2.resize(
            img,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Detect face
    faces = detector(gray, 0)

    geom_features = np.zeros(5, dtype=np.float32)
    coords = None

    if len(faces) > 0:
        # Use the largest detected face
        face = max(
            faces,
            key=lambda rect: rect.width() * rect.height()
        )

        shape = predictor(gray, face)

        landmarks = [
            (shape.part(i).x, shape.part(i).y)
            for i in range(68)
        ]

        coords = landmarks

        left_ear = calculate_ear(LEFT_EYE, landmarks)
        right_ear = calculate_ear(RIGHT_EYE, landmarks)

        ear = (left_ear + right_ear) / 2.0
        mar = calculate_mar(MOUTH, landmarks)

        pitch, yaw, roll = get_head_pose(
            landmarks,
            gray.shape
        )

        geom_features[0] = ear
        geom_features[1] = mar
        geom_features[2] = pitch
        geom_features[3] = yaw
        geom_features[4] = roll

    # Image input for the neural network
    img_norm = cv2.resize(
        img_rgb,
        (224, 224),
        interpolation=cv2.INTER_AREA
    ).astype(np.float32) / 255.0

    return img_norm, geom_features, coords


# BASE64 IMAGE DECODING

def base64_to_image(base64_string):
    if not isinstance(base64_string, str) or not base64_string:
        return None

    # Remove data URL prefix if provided
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]

    try:
        img_data = base64.b64decode(
            base64_string,
            validate=True
        )

        np_arr = np.frombuffer(
            img_data,
            dtype=np.uint8
        )

        img = cv2.imdecode(
            np_arr,
            cv2.IMREAD_COLOR
        )

        return img

    except (ValueError, TypeError):
        return None


# RESET TEMPORAL STATE

def reset_detection_state():
    global closed_frames

    ear_history.clear()
    mar_history.clear()
    closed_frames = 0


# WEB ROUTES

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "dlib_loaded": detector is not None and predictor is not None,
        "tensorflow_loaded": model_hybrid is not None
    })


# PREDICTION API

@app.route("/predict", methods=["POST"])
def predict():
    global closed_frames

    start_time = time.perf_counter()

    data = request.get_json(silent=True)

    if not data or "image" not in data:
        return jsonify({
            "error": "No image provided"
        }), 400

    img = base64_to_image(data["image"])

    if img is None:
        return jsonify({
            "error": "Invalid image"
        }), 400

    if detector is None or predictor is None:
        return jsonify({
            "error": "Dlib model is not loaded"
        }), 503

    if model_hybrid is None:
        return jsonify({
            "error": "TensorFlow model is not loaded"
        }), 503

    try:
        # Extract image and geometry features
        img_norm, geom_features, coords = extract_features(img)

        # No face detected
        if coords is None:
            reset_detection_state()

            return jsonify({
                "prediction": "No Face Detected",
                "confidence": 0.0,
                "ear": 0.0,
                "mar": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0
            })

        # Prepare model inputs
        X_img = np.expand_dims(
            img_norm,
            axis=0
        )

        X_geom = np.expand_dims(
            geom_features,
            axis=0
        )

        # Run single-frame inference
        preds = model_hybrid(
            [X_img, X_geom],
            training=False
        ).numpy()[0]

        pred_idx = int(np.argmax(preds))
        pred_class = CLASSES[pred_idx]
        confidence = float(preds[pred_idx]) * 100.0

        # Geometry features
        raw_ear = float(geom_features[0])
        raw_mar = float(geom_features[1])

        pitch = float(geom_features[2])
        yaw = float(geom_features[3])
        roll = float(geom_features[4])

        # Temporal smoothing
        ear_history.append(raw_ear)
        mar_history.append(raw_mar)

        if len(ear_history) > FRAME_WINDOW:
            ear_history.pop(0)
            mar_history.pop(0)

        smoothed_ear = float(np.mean(ear_history))
        smoothed_mar = float(np.mean(mar_history))

        # Apply EAR/MAR rules
        if smoothed_ear <= EAR_THRESHOLD:
            pred_class = "Closed"

        elif smoothed_mar > MAR_THRESHOLD:
            pred_class = "yawn"

        if (
            pred_class == "yawn"
            and smoothed_mar < 0.60
        ):
            pred_class = "no_yawn"

        # Determine final status
        if pred_class == "Closed":
            closed_frames += 1

            if closed_frames >= ALARM_THRESHOLD:
                prediction_out = "Wake Up!"
            else:
                prediction_out = "Normal"

        elif pred_class == "yawn":
            closed_frames = 0
            prediction_out = "Yawn Detected"

        else:
            closed_frames = 0
            prediction_out = "Normal"

        elapsed = time.perf_counter() - start_time

        app.logger.info(
            "Prediction completed in %.3f seconds | "
            "status=%s | EAR=%.3f | MAR=%.3f",
            elapsed,
            prediction_out,
            smoothed_ear,
            smoothed_mar
        )

        return jsonify({
            "prediction": prediction_out,
            "confidence": round(confidence, 2),
            "ear": round(smoothed_ear, 3),
            "mar": round(smoothed_mar, 3),
            "pitch": round(pitch, 2),
            "yaw": round(yaw, 2),
            "roll": round(roll, 2)
        })

    except Exception:
        app.logger.exception("Prediction failed")

        return jsonify({
            "error": "Prediction failed"
        }), 500


# ERROR HANDLERS

@app.errorhandler(413)
def request_too_large(error):
    return jsonify({
        "error": "Image request is too large"
    }), 413


# LOCAL DEVELOPMENT

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )