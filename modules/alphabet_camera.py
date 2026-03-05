import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque

# =====================
# LOAD LABELS & MODEL
# =====================

labels = ['a','b','c','d','e','f','g','h','i','j','k','l','m',
          'n','o','p','q','r','s','t','u','v','w','x','y','z']

interpreter = tf.lite.Interpreter(model_path="models/isl_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =====================
# MEDIAPIPE
# =====================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

prediction_history = deque(maxlen=10)

# =====================
# NORMALIZE
# =====================

def normalize_landmarks(landmarks, hand_label):

    pts = np.array(landmarks).reshape(21,3)

    if hand_label == "Left":
        pts[:,0] = 1.0 - pts[:,0]

    pts -= pts[0]

    return pts.flatten()

# =====================
# PREDICT FRAME
# =====================

def predict_alphabet_frame(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    current_predictions = []

    if result.multi_hand_landmarks:

        for hand_lms, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):

            hand_label = handedness.classification[0].label

            raw = [val for lm in hand_lms.landmark for val in [lm.x,lm.y,lm.z]]

            features = normalize_landmarks(raw, hand_label)

            X = np.array(features, dtype=np.float32).reshape(1,-1)

            interpreter.set_tensor(input_details[0]['index'], X)
            interpreter.invoke()

            probs = interpreter.get_tensor(output_details[0]['index'])[0]

            idx = np.argmax(probs)

            conf = probs[idx]

            current_predictions.append({
                'label': labels[idx],
                'conf': conf,
                'hand': hand_label
            })

            mp_draw.draw_landmarks(
                frame,
                hand_lms,
                mp_hands.HAND_CONNECTIONS
            )

        final_char = "?"

        if len(current_predictions) == 2:

            labels_detected = [p['label'] for p in current_predictions]

            if ('c' in labels_detected and 'i' in labels_detected) or ('d' in labels_detected):
                final_char = "d"
            else:
                best_pred = max(current_predictions, key=lambda x: x['conf'])
                final_char = best_pred['label']

        elif len(current_predictions) == 1:

            pred = current_predictions[0]

            if pred['conf'] > 0.65:
                final_char = pred['label']

        prediction_history.append(final_char)

    else:
        prediction_history.append("Waiting")

    stable_sign = max(set(prediction_history), key=prediction_history.count)

    return frame, stable_sign