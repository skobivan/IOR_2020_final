import pymurapi as mur
import time
import cv2 as cv
import math

class PD(object):
    _kp = 0.0
    _kd = 0.0
    _prev_error = 0.0
    _timestamp = 0

    def __init__(self, kp, kd):
        self._kp = kp
        self._kd = kd

    def set_kp(self, value):
        self._kp = value

    def set_kd(self, value):
        self._kd = value

    def procces(self, error):
        timestamp = time.time()
        output = self._kp * error + self._kd * (timestamp - self._timestamp) * (error - self._prev_error)
        self._timestamp = timestamp
        self._prev_error = error
        return output

def clamp(value, minimum, maximum):
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value

def clamp_motor_speed(value):
    return clamp(value, -100, 100)

def clamp_angle(angle):
    if angle > 180:
        return angle - 360
    if angle < -180:
        return angle + 360
    return angle


class Camera:
    def __init__(self):
        self.curr_image = None

    def show(self):
        if self.curr_image is not None:
            cv.imshow('img', self.curr_image)
            cv.waitKey(1)

    def update_img(self , img):
        self.curr_image = img

class CameraBottom(Camera):
    def __init__(self):    
        self.hsv_mask_min_red = (0, 50, 50)
        self.hsv_mask_max_red = (15, 255, 255)

        self.hsv_mask_min_green = (40, 40, 20)
        self.hsv_mask_max_green = (80, 255, 255)   
        
        self.hsv_mask_min_violet = (130, 10, 0)
        self.hsv_mask_max_violet = (150, 255, 240)
        
        super(CameraBottom, self).__init__()
    
    def detect_basket(self):
        copy_img = self.curr_image.copy()
        hsv_img = cv.cvtColor(self.curr_image, cv.COLOR_BGR2HSV)
        mask1 = cv.inRange(hsv_img, self.hsv_mask_min_green, self.hsv_mask_max_green)
        mask2 = cv.inRange(hsv_img, self.hsv_mask_min_red, self.hsv_mask_max_red)
        mask3 = cv.inRange(hsv_img, self.hsv_mask_min_violet, self.hsv_mask_max_violet)

        mask = mask1 + mask2 + mask3   

        cnt, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        answer = False
        coords = (0, 0)
        if cnt:
            for c in cnt:
                area = cv.contourArea(c)
                if abs(area) < 1000:
                    continue
                ((x, y), (w, h), angle) = cv.minAreaRect(c)
                answer = True
                coords = (int(x), int(y))
                cv.drawContours(copy_img, [c], 0, (255, 255, 255), 3)
                cv.circle(copy_img, coords, 5, (0, 0, 0), 2)
        
        cv.imshow('detect_basket', mask)
        cv.waitKey(5)
        
        return answer, coords

    def detect(self):
        copy_img = self.curr_image.copy()
        hsv_img = cv.cvtColor(self.curr_image, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv_img, self.hsv_mask_min_red, self.hsv_mask_max_red)

        cnt, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        answer = 'None'
        coords = (0, 0)
        if cnt:
            for c in cnt:
                area = cv.contourArea(c)
                if abs(area) < 300:
                    continue
                hull = cv.convexHull(c)
                approx = cv.approxPolyDP(hull, cv.arcLength(c, True) * 0.02, True)
                ((x, y), (w, h), angle) = cv.minAreaRect(approx)
                if len(approx) > 6:
                    cv.drawContours(copy_img, [c], 0, (255, 0, 0), 3)
                    answer = 'Circle'
                elif 3 < len(approx) < 6:
                    cv.drawContours(copy_img, [c], 0, (0, 255, 0), 3)
                    answer = 'Square'
                else:
                    cv.drawContours(copy_img, [c], 0, (255, 255, 255), 3)
                    answer = 'Triangle'
                coords = (x, y)
        cv.imshow('detect', copy_img)
        cv.waitKey(5)
        return answer, coords

class Robot(object):

    def __init__(self, yaw_p, yaw_d, depth_p, depth_d):
        self.auv = mur.mur_init()
        self.yaw_pd = PD(yaw_p, yaw_d)
        self.depth_pd = PD(depth_p, depth_d)
        self.end = False

        self.bottom_cam = CameraBottom()
        
        self.state = 0
        self.yaw = 0
        self.depth = 2
        self.speed = 0
        self.circles = 0
        self.is_tr = False

    def is_depth_stable(self, val):
        return abs(self.auv.get_depth()-val) <= 1e-2

    def is_yaw_stable(self, val):
        return abs(self.auv.get_yaw()-val) <= 1.

    def keep_yaw(self, yaw, speed):
            error = self.auv.get_yaw() - yaw
            error = clamp_angle(error)

            output = self.yaw_pd.procces(error)
            output = clamp_motor_speed(output)

            self.auv.set_motor_power(0, clamp_motor_speed(speed - output))
            self.auv.set_motor_power(1, clamp_motor_speed(speed + output))

    def keep_depth(self, depth):
        error = self.auv.get_depth() - depth
        output = self.depth_pd.procces(error)
        output = clamp_motor_speed(output)
        self.auv.set_motor_power(2, output)
        self.auv.set_motor_power(3, output)
    
    def stop_yaw(self):
        self.auv.set_motor_power(0, -30)
        self.auv.set_motor_power(1, -30)
        time.sleep(0.7)
        self.auv.set_motor_power(0, 0)
        self.auv.set_motor_power(1, 0)

    def center_basket(self):
        
        (ans, (x, y)) = self.bottom_cam.detect_basket()
        if ans:
            self.stop_yaw()
            error = 100.0
            while not 110 < y < 130:
                self.bottom_cam.update_img(self.auv.get_image_bottom())
                (ans, (x, y)) = self.bottom_cam.detect_basket()
                try:
                    error = math.atan(float(y - 120) / float(160 - x)) / math.pi * 180
                except:
                    error = -90
                error -= 90
                if error < -90:
                    error = error + 180
                if abs(error) < 5.:
                    self.speed = 20
                self.yaw = self.auv.get_yaw() - error
                self.keep_yaw(self.yaw, self.speed)
                self.keep_depth(self.depth)
            self.stop_yaw()
            return True
        else:
            self.yaw = self.auv.get_yaw()
            return False

    def go_to_pinger(self, id):
        time.sleep(2)
        error, _ = self.auv.get_pinger_data(id)
        self.yaw = clamp_angle(self.auv.get_yaw() + error)
        while not self.is_yaw_stable(self.yaw):
            self.keep_depth(self.depth)
            self.keep_yaw(self.yaw, 0)
            time.sleep(0.03)
            
        print("Find pinger ", self.yaw)
        is_it_that_pinger = True
        
        while True:
            time.sleep(0.03)
#            ans = False
            self.bottom_cam.update_img(self.auv.get_image_bottom())
            ans, _ = self.bottom_cam.detect_basket()
            # print('Ans:', ans, 'flg:', is_it_that_pinger)
            if ans and is_it_that_pinger:
                self.stop_yaw()
                time.sleep(2)
                _, dst = self.auv.get_pinger_data(id)
                print(dst)
                if dst < 2.5:
                    self.center_basket()
                    break
                else:
                    is_it_that_pinger = False
            elif not ans:
                is_it_that_pinger = True
                
            self.keep_yaw(self.yaw, self.speed)
            self.keep_depth(self.depth)

    def logic(self):
        #self.bottom_cam.update_img(self.auv.get_image_bottom())

        if self.state == 0:
            for pinger_id in range(4):
                print('State 0 pinger ', pinger_id)
                self.go_to_pinger(pinger_id)
                ans, _ = self.bottom_cam.detect()
                if ans == 'Circle':
                    self.auv.drop()
                    self.circles += 1
                elif ans == 'Triangle':
                    self.tr_id = pinger_id
                    self.is_tr = True
                if self.circles == 2 and self.is_tr:
                    break
            self.state += 1
            self.stop_yaw()
            
        if self.state == 1:
            print('State 1 triangle id ', self.tr_id)
            if self.tr_id != 3:
                self.go_to_pinger(self.tr_id)
            self.state += 1
            self.depth = 0
            self.speed = 3
        self.keep_yaw(self.yaw, self.speed)
        self.keep_depth(self.depth)

KP_YAW = 0.4
KD_YAW = 40

KP_DEPTH = 20
KD_DEPTH = 100

SPEED = 20

robot = Robot(KP_YAW, KD_YAW, KP_DEPTH, KD_DEPTH)
robot.depth = 2
robot.speed = SPEED
time.sleep(2)
while True:
    # robot.keep_depth(2)
    # robot.keep_yaw(50, 0)
    robot.logic()
    
