from urllib import response
import pyfirmata
import numpy as np
import requests
import time 
import cv2
import math
import RPi.GPIO as GPIO

#서보 모터 돌리기 위한 함수 클래스
class Servo:
    def __init__(self, ch):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(ch, GPIO.OUT)
        
        self.servo = GPIO.PWM(ch, 50)
        self.servo.start(0)
        
    def Activate(self, duty):
        self.servo.ChangeDutyCycle(duty)
        time.sleep(0.15)
    def angle_to_percent(self, angle) :
        if angle > 180 or angle < 0 :
            return False
        start = 4
        end = 12.5
        ratio = (end - start)/180 #Calcul ratio from angle to percent
        angle_as_percent = angle * ratio
        return start + angle_as_percent    
    
    def Clean(self):
        GPIO.cleanup()
        
# 라즈베리파이에서 아두이노 제어하기 위해 pyfirmata 라이브러리 사용
board = pyfirmata.Arduino('/dev/ttyACM0')

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

TRIG = 23
ECHO = 24
print("초음파 거리 측정기")

GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

GPIO.output(TRIG, False)
print("초음파 출력 초기화")
time.sleep(2)

motor1 = board.get_pin("d:4:o")
motor2 = board.get_pin("d:3:o")
ena = board.get_pin("d:5:p")


def dcstart():
    motor1.write(1)
    motor2.write(0) 

def dcstop():
    motor1.write(0)
    motor2.write(0)
def dcback():
    motor1.write(0)
    motor2.write(1)
    
#초음파 거리측정 함수    
def sonic():
    distance = 0
    while True:
        GPIO.output(TRIG,True)
        time.sleep(0.00001)        # 10uS의 펄스 발생을 위한 딜레이
        GPIO.output(TRIG, False)
      
        while GPIO.input(ECHO)==0:
            start = time.time()     # Echo핀 상승 시간값 저장
                
        while GPIO.input(ECHO)==1:
            stop = time.time()      # Echo핀 하강 시간값 저장
                
        check_time = stop - start
        distance = check_time * 34300 / 2
        print("Distance : %.1f cm" % distance)
        break
    return distance
    
servo = Servo(18)

#pc에서 각도 값 받아오기 위한 함수
def servodata():
    requestUrl = 'http://192.168.0.163:5000/servo_'
    response = requests.get(requestUrl)
    if response.status_code == 200:
        # print(response.text)
        return response.text
    return 0

dgr = servo.angle_to_percent(90)
servo.Activate(dgr)
ena.write(0.9)
dcstart()
count = 1

while True:
    if(count==0):
        dgr = servo.angle_to_percent(90)
        servo.Activate(dgr)
        ena.write(1)
        dcstart()
        time.sleep(0.5)
        ena.write(0.8)
        
    two = sonic()
    if two>35:
        ena.write(0.9)
        dcstart()
    elif two<=35:
        print('장애물감지')
        dcstop()
        servo.Activate(servo.angle_to_percent(90))
        ena.write(1)
        dcback()
        time.sleep(2)
        dcstart()
        servo.Activate(servo.angle_to_percent(112))
        time.sleep(1.8)
        servo.Activate(servo.angle_to_percent(75))
        time.sleep(1.5)
        dcstop()
        time.sleep(2)
        count = 0
        continue
    
    v = servodata()
    print(v)
    # 차선이 인식되지 않은 경우
    if v=='stop':
        print('no line')
        dcstop()
        time.sleep(2)
        servo.Activate(servo.angle_to_percent(90))
        ena.write(1)
        dcback()
        time.sleep(1.3)
        servo.Activate(servo.angle_to_percent(110))
        ena.write(1)
        dcstart()
        time.sleep(1)
        dcstop()
        count = 0
        continue
    #정지선 인식시 3초후 출발
    if v=='stopline':
        print(v)
        dcstop()
        time.sleep(3)
        dcstart()
    if v=='stop':
        print(v)
        dcstop()
        time.sleep(3)
        dcstart()
    # 바퀴의 중앙을 서보모터 90도로 설정 // 좌회전시 -각도를 전달받기 때문에 90도에서 절대값으로 더해줌 
    if int(v)<=0:
        servodgr = 90+abs(int(v))
        dgr = servo.angle_to_percent(servodgr)
        dgr = round(dgr, 2)
        servo.Activate(dgr)
        time.sleep(0.01)
        print(dgr)
    else:
        servodgr = 90-int(v)
        dgr = servo.angle_to_percent(servodgr)
        dgr = round(dgr, 2)
        servo.Activate(dgr)
     
        time.sleep(0.01)
        print(dgr)
        
    count = count+1
    
    key = cv2.waitKey(1)
    if key == 27:
        servo.Clean()
        del servo
        board.exit()
        break
    
