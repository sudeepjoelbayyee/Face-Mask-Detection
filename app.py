import cv2
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image
import streamlit as st

model = load_model('model.h5')
haar = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')
def detect_face(img):
    coods = haar.detectMultiScale(img)
    return coods

def detect_face_mask(img):
    face_img = cv2.resize(img, (224, 224))
    # Preprocess for the mask detection model
    face_img = preprocess_input(face_img.reshape(1, 224, 224, 3))
    y_pred = model.predict(face_img.reshape(1,224,224,3))
    return 0 if y_pred[0][0] <= 0.5 else 1

def draw_label(img,text,pos,bg_color):
    text_size = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,cv2.FILLED)
    and_x = pos[0] + text_size[0][0] + 2
    and_y = pos[1] + text_size[0][1] - 2

    cv2.rectangle(img,pos,(and_x,and_y),bg_color,cv2.FILLED)
    cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1,cv2.LINE_AA)

def main():
    st.set_page_config(page_title="Face Mask Detection Streamlit App", page_icon="🎭", layout="wide")
    st.sidebar.title("Face Mask Detection Streamlit App")
    camera_feed = st.empty()
    cap = cv2.VideoCapture(0)
    while True:
        ret,frame = cap.read()

        # call detection method
        img = cv2.resize(frame,(224,224))
        y_pred = detect_face_mask(img)

        coods = detect_face(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        for x, y, width, height in coods:
            face_region = frame[y:y + height, x:x + width]

            # Call mask detection on the face region
            mask_result = detect_face_mask(face_region)

            # Draw rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)

            mask_result_str = str(mask_result)

            # Draw label based on mask detection result
            cv2.putText(frame, mask_result_str, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if y_pred >= 0.5:
            draw_label(frame,'No Mask',(30,30),(0,255,0))
        else:
            draw_label(frame, 'Mask', (30, 30), (255,0, 0))

        camera_feed.image(frame, channels="BGR", width=500)

if __name__ == "__main__":
    main()