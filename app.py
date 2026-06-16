import os
import cv2
import numpy as np
import tensorflow as tf
import dlib
import base64
import urllib.request
import bz2
import shutil
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

MODEL_PATH = "best_model_fold_5.keras"
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

def download_landmark():
    if os.path.exists(PREDICTOR_PATH):
        print("Landmark model already exists.")
        return

    print("Downloading shape_predictor_68_face_landmarks.dat ...")

    url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"

    compressed_file = PREDICTOR_PATH + ".bz2"

    urllib.request.urlretrieve(url, compressed_file)

    print("Extracting landmark model...")

    with bz2.BZ2File(compressed_file, "rb") as source:
        with open(PREDICTOR_PATH, "wb") as target:
            shutil.copyfileobj(source, target)

    os.remove(compressed_file)

    print("Landmark model downloaded successfully.")

download_landmark()

print("Loading Dlib and TensorFlow models...")
try:
    detector = dlib.get_frontal_face_detector()
except Exception as e:
    print(f"Warning: Failed to initialize face detector: {e}")
    detector = None

try:
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
except Exception as e:
    print(f"Warning: Failed to load {PREDICTOR_PATH}: {e}")
    predictor = None

try:
    model_hybrid = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Warning: Failed to load {MODEL_PATH}: {e}")
    model_hybrid = None

CLASSES = ['Closed', 'Open', 'no_yawn', 'yawn']
LEFT_EYE = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]
MOUTH = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]

FRAME_WINDOW = 3
ear_history = []
mar_history = []
closed_frames = 0
ALARM_THRESHOLD = 2
EAR_THRESHOLD = 0.17
MAR_THRESHOLD = 0.70

def calculate_ear(eye_points, landmarks):
    v1 = np.linalg.norm(np.array(landmarks[eye_points[1]]) - np.array(landmarks[eye_points[5]]))
    v2 = np.linalg.norm(np.array(landmarks[eye_points[2]]) - np.array(landmarks[eye_points[4]]))
    h = np.linalg.norm(np.array(landmarks[eye_points[0]]) - np.array(landmarks[eye_points[3]]))
    return (v1 + v2) / (2.0 * h) if h != 0 else 0

def calculate_mar(mouth_points, landmarks):
    v1 = np.linalg.norm(np.array(landmarks[51]) - np.array(landmarks[59]))
    v2 = np.linalg.norm(np.array(landmarks[52]) - np.array(landmarks[58]))
    v3 = np.linalg.norm(np.array(landmarks[53]) - np.array(landmarks[57]))
    h = np.linalg.norm(np.array(landmarks[48]) - np.array(landmarks[54]))
    return (v1 + v2 + v3) / (3.0 * h) if h != 0 else 0

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
        landmarks[30], landmarks[8], landmarks[36],
        landmarks[45], landmarks[48], landmarks[54]
    ], dtype="double")

    focal_length = img_size[1]
    center = (img_size[1]/2, img_size[0]/2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4,1))

    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    if not success: return 0, 0, 0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rotation_matrix, translation_vector))
    euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

    pitch = euler_angles[0][0]
    yaw = euler_angles[1][0]
    roll = euler_angles[2][0]
    return pitch, yaw, roll

def extract_features(img, augment=False):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)

    geom_features = np.zeros(5, dtype=np.float32)
    coords = None

    if len(faces) > 0 and predictor is not None:
        face = faces[0]
        shape = predictor(gray, face)
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        coords = landmarks

        left_ear = calculate_ear(LEFT_EYE, landmarks)
        right_ear = calculate_ear(RIGHT_EYE, landmarks)
        geom_features[0] = (left_ear + right_ear) / 2.0

        geom_features[1] = calculate_mar(MOUTH, landmarks)

        pitch, yaw, roll = get_head_pose(landmarks, gray.shape)
        geom_features[2] = pitch
        geom_features[3] = yaw
        geom_features[4] = roll

    img_norm = cv2.resize(img_rgb, (224, 224)) / 255.0
    return img_norm, geom_features, coords

def base64_to_image(base64_string):
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    img_data = base64.b64decode(base64_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global ear_history, mar_history, closed_frames
    
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
        
    img = base64_to_image(data['image'])
    img_norm, geom_features, coords = extract_features(img, augment=False)
    
    if coords is None:
        return jsonify({
            "prediction": "No Face Detected",
            "confidence": 0.0,
            "ear": 0.0,
            "mar": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0
        })
        
    X_img = np.expand_dims(img_norm, axis=0)
    X_geom = np.expand_dims(geom_features, axis=0)
    
    if model_hybrid is None:
        pred_class = "Open"
        confidence = 0.0
    else:
        preds = model_hybrid.predict([X_img, X_geom], verbose=0)[0]
        pred_idx = np.argmax(preds)
        pred_class = CLASSES[pred_idx]
        confidence = float(preds[pred_idx]) * 100
        
    raw_ear = float(geom_features[0])
    raw_mar = float(geom_features[1])
    pitch = float(geom_features[2])
    yaw = float(geom_features[3])
    roll = float(geom_features[4])
    
    ear_history.append(raw_ear)
    mar_history.append(raw_mar)
    if len(ear_history) > FRAME_WINDOW:
        ear_history.pop(0)
        mar_history.pop(0)
        
    smoothed_ear = float(np.mean(ear_history))
    smoothed_mar = float(np.mean(mar_history))
    
    if smoothed_ear <= EAR_THRESHOLD:
        pred_class = 'Closed'
    elif smoothed_mar > MAR_THRESHOLD:
        pred_class = 'yawn'
        
    if (pred_class == 'yawn' or (pred_class == 'Closed' and smoothed_ear > EAR_THRESHOLD)) and smoothed_mar < 0.60:
        pred_class = 'no_yawn'

    if pred_class == 'Closed':
        closed_frames += 1
        if closed_frames >= ALARM_THRESHOLD:
            prediction_out = "Wake Up!"
        else:
            prediction_out = "Normal"
    elif pred_class == 'yawn':
        closed_frames = 0
        prediction_out = "Yawn Detected"
    else:
        closed_frames = 0
        prediction_out = "Normal"

    return jsonify({
        "prediction": prediction_out,
        "confidence": round(confidence, 2),
        "ear": round(smoothed_ear, 3),
        "mar": round(smoothed_mar, 3),
        "pitch": round(pitch, 2),
        "yaw": round(yaw, 2),
        "roll": round(roll, 2)
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
