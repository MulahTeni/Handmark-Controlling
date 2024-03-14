import cv2
import pyautogui
import mediapipe as mp
import os
import numpy as np
import csv

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.6, min_tracking_confidence=0.9)

mp_drawing = mp.solutions.drawing_utils

def save_gest_to_csv(arr, label):
    file_name = "labeled_gestures.csv"
    arr.extend([label])
    with open(file_name, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(arr)
        print(f"Saved: {arr} label: {label}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            xy_coords = []
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                xy_coords.extend([x, y])

            key = cv2.waitKey(5)
            if 49 <= key <= 53:
                label = key - 49
                save_gest_to_csv(xy_coords, [label])

    cv2.imshow('Hand Gesture', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
