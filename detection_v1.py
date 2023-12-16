import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, render_template, Response

#Carregar o modelo treinado
model = load_model('/home/eenjp/Área de trabalho/Undergraduate/2023.2/Visão Computacional/Trabalhos/Models/sign_language_model.h5')

class VideoCamera(object):
    def __init__(self):
        #Câmera
        self.cap = cv2.VideoCapture(0)
        #Coordenadas da ROI
        self.roi_x, self.roi_y, self.roi_width, self.roi_height = 40, 100, 200, 200

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
            #Captura
            ret, frame = self.cap.read()

            frame = cv2.flip(frame, 1)

            #ROI
            roi = frame[self.roi_y:self.roi_y+self.roi_height, self.roi_x:self.roi_x+self.roi_width]

            img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

            #Previsão usando o modelo
            y_pred = self.recognize(img)

            #Caractere equivalente
            char_op = chr(y_pred + 65)

            #Desenhar a ROI no frame
            cv2.rectangle(frame, (self.roi_x, self.roi_y), (self.roi_x+self.roi_width, self.roi_y+self.roi_height), (255, 0, 0), 2)

            #Exiba o caractere equivalente na ROI
            cv2.putText(frame, char_op, (self.roi_x + 10, self.roi_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            #Frame bruto em jpg
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#Inicializr o Flask
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
