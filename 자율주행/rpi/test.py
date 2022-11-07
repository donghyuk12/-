from urllib import response
import numpy as np
import requests
import time 
import cv2
import math
import RPi.GPIO as GPIO
import wiringpi

# 모터 상태
STOP  = 0
START  = 1
BACK = 2

# 모터 채널
CH1 = 0

# PIN 입출력 설정
OUTPUT = 1
INPUT = 0

# PIN 설정
HIGH = 1
LOW = 0

# 실제 핀 정의
#PWM PIN
ENA = 25

#GPIO PIN
IN1 = 24
IN2 = 23
def setPinConfig(EN, INA, INB):
    wiringpi.pinMode(EN, OUTPUT)
    wiringpi.pinMode(INA, OUTPUT)
    wiringpi.pinMode(INB, OUTPUT)
    wiringpi.softPwmCreate(EN, 0, 255)

# 모터 제어 함수
def setMotorContorl(PWM, INA, INB, speed, stat):
    #모터 속도 제어 PWM
    wiringpi.softPwmWrite(PWM, speed)

    #앞으로
    if stat == START:
        wiringpi.digitalWrite(INA, HIGH)
        wiringpi.digitalWrite(INB, LOW)
    #뒤로
    elif stat == BACK:
        wiringpi.digitalWrite(INA, LOW)
        wiringpi.digitalWrite(INB, HIGH)
    #정지
    elif stat == STOP:
        wiringpi.digitalWrite(INA, LOW)
        wiringpi.digitalWrite(INB, LOW)

# 모터 제어함수 간단하게 사용하기 위해 한번더 래핑(감쌈)
def setMotor(speed, stat):
    setMotorContorl(ENA, IN1, IN2, speed, stat)
def Clean(self):
        GPIO.cleanup()
#GPIO 라이브러리 설정
wiringpi.wiringPiSetup()

#모터 핀 설정
setPinConfig(ENA, IN1, IN2)

#제어 시작
def dcstart(v):
    setMotor(v, START)

def dcback(v):
    setMotor(v, BACK)    

def dcstop():
    setMotor(0, STOP)
    wiringpi.delay(100)

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
        
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

TRIG = 27
ECHO = 22
print("초음파 거리 측정기")

GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

GPIO.output(TRIG, False)
print("초음파 출력 초기화")
time.sleep(2)
    
def sonic():
    #초음파 거리측정 함수
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
                

def servodata():
    requestUrl = 'http://192.168.0.163:5000/servo_'
    response = requests.get(requestUrl)
    if response.status_code == 200:
        # print(response.text)
        return response.text
    return 0

# move_servo(96)
if __name__ == '__main__':
 
    servo = Servo(18)
    try:
        dgr = servo.angle_to_percent(90)
        servo.Activate(dgr)
        time.sleep(1)
        while True:
            dcstart(250)
            # two = sonic()
            
            # if two<=35:
            #     print('장애물감지')
            #     dcstop()
            #     servo.Activate(servo.angle_to_percent(90))
            #     dcback(180)
            #     time.sleep(2)
            #     dcstart(200)
            #     servo.Activate(servo.angle_to_percent(112))
            #     # ena.write(1)
            #     # dcstart()
            #     time.sleep(1.8)
            #     servo.Activate(servo.angle_to_percent(75))
            #     time.sleep(1.5)
            #     dcstop(150)
            #     time.sleep(2)
            #     count = 0
            #     continue
            
            v = servodata()
            if int(v)<=0:
                servodgr = 90+abs(int(v))
                dgr = servo.angle_to_percent(servodgr)
                dgr = round(dgr, 2)
                servo.Activate(dgr)
                time.sleep(0.1)
                print(dgr)
            else:
                servodgr = 90-int(v)
                dgr = servo.angle_to_percent(servodgr)
                dgr = round(dgr, 2)
                servo.Activate(dgr)
                time.sleep(0.1)
                print(dgr)

            if v=='stop':
                print('no line')
                dcstop()
                time.sleep(1)
                servo.Activate(servo.angle_to_percent(90))
                dcback(180)
                time.sleep(1)
                continue
            
            if v=='stopline':
                print(v)
                dcstop()
                time.sleep(3)
            if v=='stoplane':
                print(v)
                dcstop()
            if v=='stoplines':
                dcstop()
                print(v)
                time.sleep(3)
    except:
        servo.Clean()
        del servo   
        Clean()
        # two = sonic() #초음파 감지
            
        #     time.sleep(1.2)
            
            
        # if v=='stoplane':
        #     print('stoplane')
        #     dcstop()
        #     servo.Clean()
        #     del servo
        # if v=='stopline':
        #     print('stopline')
        #     dcstop()
        #     break
        # ena.write(1)
        # # dcstart()
        # time.sleep(0.7)
        # dcstop()
        
    
