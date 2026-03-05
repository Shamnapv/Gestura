import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque

# Load model
model = load_model("models/sign_language_lstm_improved_gaussian.h5")

actions = np.array(['good', 'hello', 'iloveyou', 'please', 'thankyou', 'yes', 'you'])

# Buffers
sequence = []
sentence = []
threshold = 0.85

prob_buffer = deque(maxlen=5)
pred_buffer = deque(maxlen=10)

# Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def extract_keypoints(results):

    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21*3)

    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([lh,rh])


def draw_probability_bars(image, avg_res):

    for num,(action,prob) in enumerate(zip(actions,avg_res)):

        cv2.rectangle(image,(640,60+num*40),(760,90+num*40),(200,200,200),-1)

        cv2.rectangle(image,(640,60+num*40),
                      (640+int(prob*120),90+num*40),
                      (245,117,16),-1)

        cv2.putText(image,action,(645,83+num*40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,(0,0,0),1,cv2.LINE_AA)

    return image


def predict_word():

    global sequence, sentence

    cap = cv2.VideoCapture(0)

    cv2.namedWindow("Real-time Sign Language Translation")

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while cap.isOpened():

            ret,frame = cap.read()

            if not ret:
                break

            image,results = mediapipe_detection(frame,holistic)

            mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

            mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

            keypoints = extract_keypoints(results)

            sequence.append(keypoints)

            sequence = sequence[-30:]

            if len(sequence)==30:

                res = model.predict(
                    np.expand_dims(sequence,axis=0),
                    verbose=0
                )[0]

                prob_buffer.append(res)

                avg_res = np.mean(prob_buffer,axis=0)

                pred_index = np.argmax(avg_res)

                pred_buffer.append(pred_index)

                consistent = pred_buffer.count(pred_index) >= 8

                if avg_res[pred_index] > threshold and consistent:

                    if len(sentence) > 0:

                        if actions[pred_index] != sentence[-1]:

                            sentence.append(actions[pred_index])

                    else:

                        sentence.append(actions[pred_index])

                if len(sentence)>5:
                    sentence = sentence[-5:]

                draw_probability_bars(image,avg_res)

            cv2.rectangle(image,(0,0),(640,40),(245,117,16),-1)

            cv2.putText(image,' '.join(sentence),(3,30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,(255,255,255),2,cv2.LINE_AA)

            cv2.imshow("Real-time Sign Language Translation",image)

            key = cv2.waitKey(10) & 0xFF

            if key==ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    return " ".join(sentence)