import cv2
import os

DATASET_PATH="dataset"

def show_word_video(word):

    folder = os.path.join(DATASET_PATH,word.lower())

    if not os.path.exists(folder):
        return

    videos = os.listdir(folder)

    path = os.path.join(folder,videos[0])

    cap = cv2.VideoCapture(path)

    while cap.isOpened():

        ret,frame = cap.read()

        if not ret:
            break

        cv2.imshow("Word Sign",frame)

        if cv2.waitKey(25)==ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()