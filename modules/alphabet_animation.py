import cv2
import os
import time

DATASET_PATH="data"

def show_letter(letter):

    if letter==" ":
        time.sleep(1)
        return

    path=os.path.join(DATASET_PATH,letter.upper())

    images=os.listdir(path)

    img=cv2.imread(os.path.join(path,images[0]))

    cv2.imshow("Alphabet Sign",img)

    cv2.waitKey(1200)

    cv2.destroyAllWindows()