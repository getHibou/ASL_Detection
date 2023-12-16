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

        #Coordenadas do box fixo
        self.box_x, self.box_y, self.box_width, self.box_height = 100, 100, 200, 200

    def __del__(self):
        #Recursos da câmera
        self.cap.release()

    def recognize(self, img):
        img = np.resize(img, (28, 28, 1))
        img = np.expand_dims(img, axis=0)
        img = np.asarray(img)
        classes = model.predict(img)[0]
        pred_id = list(classes).index(max(classes))
        prob = max(classes)
        return pred_id, prob

    def gen_frame(self):
        while True:
            #Frame
            ret, frame = self.cap.read()

            #Inverter horizontalmente o frame
            frame = cv2.flip(frame, 1)

            #Criar uma imagem preta do mesmo tamanho que o frame
            black_frame = np.zeros_like(frame)

            #Defina a ROI dentro da imagem preta
            roi_black = black_frame[self.box_y:self.box_y + self.box_height, self.box_x:self.box_x + self.box_width]

            #Copiar a ROI da imagem original para a imagem preta
            roi_black[:, :] = frame[self.box_y:self.box_y + self.box_height, self.box_x:self.box_x + self.box_width]

            #Processamento da ROI para detecção de linguagem de sinais
            roi_gray = cv2.cvtColor(roi_black, cv2.COLOR_BGR2GRAY)
            roi_resized = cv2.resize(roi_gray, (28, 28), interpolation=cv2.INTER_AREA)

            #Previsão usando o modelo
            pred_id, prob = self.recognize(roi_resized)

            #Caractere equivalente
            char_op = chr(pred_id + 65)

            #Exiba a letra e a probabilidade dentro do box em preto
            cv2.putText(black_frame, f"{char_op} - {prob:.2f}", (self.box_x, self.box_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            #Frame bruto em jpg
            ret, jpeg = cv2.imencode('.jpg', black_frame)
            black_frame = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + black_frame + b'\r\n')

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
