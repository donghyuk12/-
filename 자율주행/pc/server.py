from email.mime.image import MIMEImage
from flask import Flask, request, Response
import cv2
from lane import *
from threading import Thread
app = Flask(__name__)

# 영상처리한 이미지 플라스크 서버로 웹 스트리밍 // 함수 반환값을 받을 때 이미지도 같이 받아와야함
# def gen_frames():  # generate frame by frame from camera
#     t = Thread(target=func1, daemon=True)
#     t.start()
#     while True:
#         # Capture frame-by-frame
#         success, frame = func1() # read the camera frame
#         if (not success):
#             break
#         else:
#             # cv2.resize(frame, (640, 480))
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
            
#             # curr_time = time.time()
#             # total_frames = total_frames + 1

#             # term = curr_time - prev_time
            
#             # fps = 1 / term
#             # prev_time = curr_time
#             # fps_string = f'FPS = {fps:.2f}'
#             # cv2.putText(frame, fps_string, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255))
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#             time.sleep(0.02)
            
# @app.route('/lane')
# def video_feed():
#     return Response(gen_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# 영상처리 후 각도 값 라즈베리파이로 넘겨주기 위한 함수
@app.route('/servo_')
def confirm():
    t = Thread(target=func1, daemon=True)
    t.start()
    accuracy = func1()
    return str(accuracy)



if __name__ == '__main__':
        app.run(
            host="0.0.0.0",
            port=5000,
            debug=True)
        # degree = confirm()
        