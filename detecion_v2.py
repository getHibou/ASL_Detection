import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, render_template, Response
import mediapipe as mp

#Carregar o modelo treinado
model = load_model('/home/eenjp/Área de trabalho/Undergraduate/2023.2/Visão Computacional/Trabalhos/Models/sign_language_model.h5')

class VideoCamera(object):
    def __init__(self):
        #Câmera
        self.cap = cv2.VideoCapture(0)
        #Rastreador de mãos do Mediapipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils

    def __del__(self):
        #Recursos da câmera
        self.cap.release()

    def recognize(self, img):
        img = np.resize(img, (28, 28, 1))
        img = np.expand_dims(img, axis=0)
        img = np.asarray(img)
        classes = model.predict(img)[0]
        pred_id = list(classes).index(max(classes))
        return pred_id

    def gen_frame(self):
        while True:
            #Frame
            ret, frame = self.cap.read()

            #Conversão do frame para RGB (mediapipe usa RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            #Processamento do frame para detecção de mão
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    #Extraia as coordenadas da ponta do dedo indicador
                    index_finger_x = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
                    index_finger_y = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])

                    cv2.line(frame, (int(hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x * frame.shape[1]), 
                        int(hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y * frame.shape[0])),
                            (index_finger_x, index_finger_y), (0, 255, 0), 2)


                    #ROI perto da ponta do dedo indicador
                    roi_size = 100
                    roi_x = max(0, index_finger_x - roi_size // 2)
                    roi_y = max(0, index_finger_y - roi_size // 2)
                    roi = frame[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]

                    #Processamento da ROI para detecção de linguagem de sinais
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    roi_resized = cv2.resize(roi_gray, (28, 28), interpolation=cv2.INTER_AREA)

                    # Previsão usando o modelo
                    y_pred = self.recognize(roi_resized)

                    #Caractere equivalente
                    char_op = chr(y_pred + 65)

                    #Exiba o caractere equivalente perto da ponta do dedo indicador
                    cv2.putText(frame, char_op, (index_finger_x, index_finger_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            #Frame bruto em jpg
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(VideoCamera().gen_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run()
