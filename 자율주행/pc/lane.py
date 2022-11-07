import cv2
from threading import Thread
import time
import numpy as np
import math

# 라즈베리파이에서 웹 스트리밍 하는 영상 가져오는 함수 클래스
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(320,240),framerate=30):
        self.stream = cv2.VideoCapture("http://192.168.0.143:8080/video_feed12")
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update,args=()).start()
        return self
    
    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# 이미지 템플릿 매칭
template = cv2.imread("./stop.jpg", cv2.IMREAD_GRAYSCALE)
w, h = template.shape[::-1]
TextStop='STOP'
font = cv2.FONT_HERSHEY_SIMPLEX
org = (140,100)

# 정지선 인식 함수
def stopline_(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    can = cv2.Canny(gray, 80, 240, 3, None)
    # 관심 구역 설정
    rectangle = np.array([[(80, 215), (80, 235), (180, 235), (180, 215)]],dtype=np.int32)
    mask = np.zeros_like(can)
    cv2.fillPoly(mask, rectangle, 255)
    imgmask = np.zeros_like(src)
    color = [0,0,255]
    cv2.fillPoly(imgmask, rectangle, color)
    Roi_image = cv2.bitwise_and(src,imgmask)
    asd = cv2.addWeighted(src, 1, Roi_image, 1, 0)
    masked_image = cv2.bitwise_and(can, mask)
    ccan = cv2.cvtColor(masked_image, cv2.COLOR_GRAY2BGR)
    # 수평 직선 검출   
    stop_linesP = cv2.HoughLinesP(masked_image, 1, np.pi / 2, None, minLineLength=70, maxLineGap=10)
    if stop_linesP is not None:
        for i in range(0, len(stop_linesP)):
            l = stop_linesP[i][0]
            cv2.line(ccan, (l[0], l[1]), (l[2], l[3]), (0, 255, 255), 10, cv2.LINE_AA)
            cv2.putText(asd, 'STOP', (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3, cv2.LINE_AA)
            return 'stopline'
    mimg = cv2.addWeighted(asd, 1, ccan, 1, 0)
    return mimg

#조감도 필터링 함수 // 노란색 차선 인식
def color_filter(image):
    hls = image
    dx = cv2.Sobel(image, -1, 1, 0)
    dy = cv2.Sobel(image, -1, 0, 1)
    do = cv2.bitwise_or(dx, dy)
    hls = cv2.bitwise_or(hls, do)
    lower = np.array([20, 150, 20])
    upper = np.array([255, 255, 255])

    yellow_lower = np.array([0, 85, 81])
    yellow_upper = np.array([190, 255, 255])

    yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
    # white_mask = cv2.inRange(hls, lower, upper)
    # mask = cv2.bitwise_or(yellow_mask, white_mask)
    masked = cv2.bitwise_and(image, image, mask = yellow_mask)
    return masked

#관심 영역 설정
def roi(image):
    x = int(image.shape[1])
    y = int(image.shape[0])
    _shape = np.array(
        [[0, y], [0, 0], [int(0.5*x), 0], [int(0.5*x), int(y)], [int(0.5*x), int(y)], [int(0.5*x), 0],[int(x), 0], [int(x), int(y)], [0, int(y)]])
    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, np.int32([_shape]), ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Bird eye view 변환 함수 // 하늘에서 내려다 보는 방식
def wrapping(image):
    (h, w) = (image.shape[0], image.shape[1])
    source = np.float32([[20, 200], [300, 200], [320, 240], [0,240]])
    destination = np.float32([[40, 0], [270, 0], [285, h], [50, h]])
    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    minv = cv2.getPerspectiveTransform(destination, source)
    _image = cv2.warpPerspective(image, transform_matrix, (w, h))
    return _image, minv

# 이미지의 가운데를 기준으로 왼쪽 차선과 오른쪽 차선 나누는 함수
def plothistogram(image):
    histogram = np.sum(image[0:, :], axis=0)
    midpoint = np.int32(histogram.shape[0]/2)
    leftbase = np.argmax(histogram[:midpoint])
    rightbase = np.argmax(histogram[midpoint:]) + midpoint
    return leftbase, rightbase

# wrapping한 이미지
def slide_window_search(binary_warped, left_current, right_current):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    nwindows = 9
    window_height = np.int32(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()  # 선이 있는 부분의 인덱스만 저장
    nonzero_y = np.array(nonzero[0])  # 선이 있는 부분 y의 인덱스 값
    nonzero_x = np.array(nonzero[1])  # 선이 있는 부분 x의 인덱스 값
    margin = 50
    minpix = 50
    left_lane = []
    right_lane = []
    color = [0, 255, 0]
    thickness = 2
    for w in range(nwindows):
        win_y_low = binary_warped.shape[0] - (w + 1) * window_height  # window 윗부분
        win_y_high = binary_warped.shape[0] - w * window_height  # window 아랫 부분
        win_xleft_low = left_current - margin  # 왼쪽 window 왼쪽 위
        win_xleft_high = left_current + margin  # 왼쪽 window 오른쪽 아래
        win_xright_low = right_current - margin  # 오른쪽 window 왼쪽 위
        win_xright_high = right_current + margin  # 오른쪽 window 오른쪽 아래
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), color, thickness)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), color, thickness)
        good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]
        left_lane.append(good_left)
        right_lane.append(good_right)
        if len(good_left) > minpix:
            left_current = np.int32(np.mean(nonzero_x[good_left]))
        if len(good_right) > minpix:
            right_current = np.int32(np.mean(nonzero_x[good_right]))
    
    left_lane = np.concatenate(left_lane)  # np.concatenate() -> array를 1차원으로 합침
    right_lane = np.concatenate(right_lane)
    leftx = nonzero_x[left_lane]
    lefty = nonzero_y[left_lane]
    rightx = nonzero_x[right_lane]
    righty = nonzero_y[right_lane]
   
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    ltx = np.trunc(left_fitx)  # np.trunc() -> 소수점 부분을 버림
    rtx = np.trunc(right_fitx)

    out_img[nonzero_y[left_lane], nonzero_x[left_lane]] = [255, 0, 0]
    out_img[nonzero_y[right_lane], nonzero_x[right_lane]] = [0, 0, 255]

    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color = 'yellow')
    # plt.plot(right_fitx, ploty, color = 'yellow')
    # plt.xlim(0, 640)
    # plt.ylim(480, 0)
    # plt.show()
    # cv2.imshow('out', out_img)
    ret = {'left_fitx' : ltx, 'right_fitx': rtx, 'ploty': ploty}
    return ret

# 인식한 차선의 범위에 색을 넣어주는 함수
def draw_lane_lines(original_image, warped_image, Minv, draw_info):
    left_fitx = draw_info['left_fitx']
    right_fitx = draw_info['right_fitx']
    ploty = draw_info['ploty']

    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

    pts = np.hstack((pts_left, pts_right))
    
    mean_x = np.mean((left_fitx, right_fitx), axis=0)
    pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])
    cv2.fillPoly(color_warp, np.int_([pts]), (216, 168, 74))
    cv2.fillPoly(color_warp, np.int_([pts_mean]), (216, 168, 74))
    img = cv2.line(color_warp, np.int32(pts_mean[0][0]), [160, 0], (0,0,255), 2)
    img = np.zeros((320, 240, 3), np.uint8)
    img = cv2.line(img, [160, 240], np.int32(pts_mean[0][-1]), (0,0,255), 2)
    deg = (np.int32(pts_mean[0][-1][0])-160)/240
    radian = round(math.tan(deg), 2)
    seta = math.degrees(radian)
    seta = math.trunc(seta)
    newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
    result = cv2.addWeighted(original_image, 1, newwarp, 0.4, 0)
    return pts_mean, result, seta

videostream = VideoStream(resolution=(320,240),framerate=25).start()
time.sleep(1)

def func1():
    degree = 0
    try:
        while True:
            frame = videostream.read()
            if not frame:
                break
            frame = cv2.resize(frame, (320, 240))
            frame = frame+10
            stopimg = frame
            stop = stopline_(frame)
            if stop == 'stopline':
                return 'stopline'
           
            wrapped_img, minverse = wrapping(frame)

            ## 조감도 필터링
            w_f_img = color_filter(wrapped_img)
            # cv2.imshow('w_f_img', w_f_img)

            w_f_r_img = roi(w_f_img)
            # cv2.imshow('w_f_r_img', w_f_r_img)
            
            ## 조감도 선 따기 wrapped img threshold
            _gray = cv2.cvtColor(w_f_r_img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(_gray, 160, 255, cv2.THRESH_BINARY)
            # cv2.imshow('threshold', thresh)

            ## 선 분포도 조사 histogram
            leftbase, rightbase = plothistogram(thresh)

            ## histogram 기반 window roi 영역
            draw_info, mimg = slide_window_search(thresh, leftbase, rightbase)
            
            ## 원본 이미지에 라인 넣기
            meanPts, result, seta = draw_lane_lines(frame, thresh, minverse, draw_info)
            # response = requests.get(requestUrl, data = seta)
            degree = round(seta, 1)
            
            cv2.putText(result, str(degree), (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3, cv2.LINE_AA)
            # result = handle(result)
            ccan = stop(stop)
            mimg = cv2.addWeighted(result, 0.5, ccan, 0.5, 0)
            # cv2.imshow('frame',result)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= 0.7)
            for pt in zip(*loc[::-1]):
                cv2.rectangle(stopimg, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
                cv2.putText(stopimg,TextStop, org, font, 1, (0,0,255), 3)
                return 'stop'
            mimg = cv2.addWeighted(result, 0.5, stopimg, 0.5, 0)
            # cv2.imshow('test', mimg)
            return degree
    except:
        return 'stop'
cv2.destroyAllWindows()