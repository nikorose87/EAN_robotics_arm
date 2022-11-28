import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
# cap = cv2.VideoCapture("video_0001.mp4")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1) as holistic:

    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        # rostro
        mp_drawing.draw_landmarks(
            frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(0, 128, 255), thickness=2))

        # Mano izquierda (azul)
        mp_drawing.draw_landmarks(
            frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=1),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))

        # Mano derecha (verde)
        mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1),
            mp_drawing.DrawingSpec(color=(57, 143, 0), thickness=2))

        # Postura
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(128, 0, 255), thickness=2, circle_radius=1),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))
        print("############# COORDENADAS X , Y  PUNTO 11 #########")
        print(results.pose_landmarks.landmark[11].x)
        print(results.pose_landmarks.landmark[11].y)
        print("############# COORDENADAS X , Y  PUNTO 13 #########")
        print(results.pose_landmarks.landmark[13].x)
        print(results.pose_landmarks.landmark[13].y)
        print("############# COORDENADAS X , Y  PUNTO 15 #########")
        print(results.pose_landmarks.landmark[15].x)
        print(results.pose_landmarks.landmark[15].y)
        print("###")
        frame = cv2.flip(frame, 1)
        cv2.imshow("Frame", frame)




        if cv2.waitKey(1) & 0xFF == 27:
            break


cap.release()
cv2.destroyAllWindows()