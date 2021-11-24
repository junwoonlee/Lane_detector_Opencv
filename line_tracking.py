import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import random
import os, sys


def display_file(img, size_x, size_y):
    height, width = img.shape[:2]
    print("height:" , height , "width:", width)
    
    res = cv2.resize(img, dsize=(size_x, size_y), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("image", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def grayscale(img): 
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussian_blur(img, kernel_size): 
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def ROI_Mask(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, (255,255,255))
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap): # 허프 변환
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

def draw_lines(img, lines):
    for line in lines:
        x1, y1, x2, y2 = line[0] 
        color = (0, 0, 255)
        img = cv2.line(img, (x1, y1), (x2, y2), color, 2)
    return img

def filter_vlines(lines):
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0] 
        angle_line = math.atan2(y1-y2, x1-x2)
        if (angle_line > np.pi/10 and angle_line < np.pi*9/10):
            filtered_lines.append(line)
        if (angle_line > -np.pi*9/10 and angle_line < -np.pi/10):
            filtered_lines.append(line)
    return filtered_lines

def points_Random_sampling(points):
    one = random.choice(points)
    two = random.choice(points)
    if(two[0]==one[0]): 
        while two[0]==one[0]:
            two = random.choice(points)
    one, two = one.reshape(1,2), two.reshape(1,2)
    three = np.concatenate((one,two),axis=1)
    three = three.squeeze()
    return three

def compute_model_parameter(line):
    m = (line[3] - line[1])/(line[2] - line[0])
    n = line[1] - m*line[0]
    a, b, c = m, -1, n
    par = np.array([a,b,c])
    return par

def compute_distance(par, point):
    return np.abs(par[0]*point[:,0]+par[1]*point[:,1]+par[2])/np.sqrt(par[0]**2+par[1]**2)

def model_verification(par, lines):
    distance = compute_distance(par,lines)
    sum_dist = distance.sum(axis=0)
    avg_dist = sum_dist/len(lines)
    
    return avg_dist

def get_fitline(img, parameter, offset_upper = 0, offset_lower = 0 ):
    m = parameter[0]
    n = parameter[2]
    height, width = img.shape[:2]
    y_upper = 1*height/5 + offset_upper
    y_lower = 4*height/5 - offset_lower

    x_upper = (y_upper -  n) / m
    x_lower = (y_lower -  n) / m

    return [int(x_upper), int(y_upper), int(x_lower), int(y_lower)]

def ransac_fit(img, lines, min=100):
    if(len(lines) != 0):
        for i in range(30):           
            sample = points_Random_sampling(lines)
            parameter = compute_model_parameter(sample)
            cost = model_verification(parameter, lines)                        
            if cost < min: 
                min = cost
                best_parameter = parameter
            if min < 15: break
        best_line = get_fitline(img, best_parameter, 500,  0)
        return best_line


image_raw = cv2.imread("C:/Users/JUN/Desktop/DR PR/2021.11 CV_car_driving_Project/test9.png")
height, width = image_raw.shape[:2]

vertices = np.array([[(500, height - 130),(900, height - 400), (width-900, height - 400), (width-300, height - 130)]], dtype=np.int32)
grey_img = grayscale(image_raw)
blur_Gimg = gaussian_blur(grey_img, 3)
ROI_GBimage = ROI_Mask(blur_Gimg, vertices)

edges = ROI_Mask(cv2.Canny(grey_img, 70, 100, apertureSize = 3), vertices)

lines = cv2.HoughLinesP(edges, 1, np.pi/360, 10, 0, 10)
vlines = filter_vlines(lines)

L_line = []
R_line = []

for i in range(len(vlines)):
    if(vlines[i][0][0] > width / 2):
        R_line.append([vlines[i][0]])
    else:
        L_line.append([vlines[i][0]])
    

Lpoints = np.reshape(np.squeeze(L_line), (-1,2))
Rpoints = np.reshape(np.squeeze(R_line), (-1,2))

ransac_L = ransac_fit(image_raw, Lpoints, 100)
ransac_R = ransac_fit(image_raw, Rpoints, 100)

print(ransac_L)
print(ransac_R)
Regressed_lines = [[ransac_R], [ransac_L]]
print(Regressed_lines)

tr_img = draw_lines(image_raw, Regressed_lines)
display_file(tr_img, 1080, 640)
