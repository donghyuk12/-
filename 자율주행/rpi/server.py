from flask import Flask, Response
import cv2
import time

app = Flask(__name__, template_folder='template')

camera = cv2.VideoCapture(0)  # use 0 for web camera

# 라즈베리파이 파이캠으로 찍고있는 영상을 플라스크 서버로 웹 스트리밍 하는 함수
def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if (not success):
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.04)

@app.route('/video_feed12')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame') 

if (__name__ == '__main__'):
    app.run(host='0.0.0.0', port=8080)

