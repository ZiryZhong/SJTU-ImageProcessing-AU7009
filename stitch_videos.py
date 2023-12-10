import cv2
from panorama import Panaroma
import numpy as np
import time

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

# 加载四个视频  
cap1 = cv2.VideoCapture('./data/testolabc1.avi')  
cap2 = cv2.VideoCapture('./data/testolabc2.avi')  
cap3 = cv2.VideoCapture('./data/testolabc3.avi')
cap4 = cv2.VideoCapture('./data/testolabc4.avi')

if(cap1.isOpened() == False | cap2.isOpened()==False | cap3.isOpened()==False | cap3.isOpened()==False):
    print("Opening video stream fail.")

panaroma = Panaroma()
H_list = []
C_list = []    
frame_cnt = 0
init_flag = True
cv2.namedWindow("panaroma", cv2.WINDOW_NORMAL)
cv2.resizeWindow("panaroma", 2070, 375)
while True:  
    start = time.time()
    # 读取视频帧  
    ret1, frame1 = cap1.read()  
    ret2, frame2 = cap2.read()  
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()

    if not ret1 or not ret2 or not ret3 or not ret4:  
        break  
    
    # 逐帧进行实时视频拼接
    frame1, mask = panaroma.get_pad_image(frame1,[720,640])
    frame2, mask = panaroma.get_pad_image(frame2,[720,640])
    frame3, mask = panaroma.get_pad_image(frame3,[720,640])
    frame4, mask = panaroma.get_pad_image(frame4,[720,640])
    if init_flag:
        # # 由第一帧确定变换参数和最佳拼接线
        H, C = panaroma.get_video_H_and_best_seam([frame3,frame4],mask,lowe_ratio=0.51)
        H_list.append(H), C_list.append(C)
        frame34 = panaroma.image_stitch_for_video([frame3,frame4],H,C)
        
        H, C = panaroma.get_video_H_and_best_seam([frame1,frame2],mask)
        H_list.append(H), C_list.append(C)
        frame12 = panaroma.image_stitch_for_video([frame1,frame2],H,C)
        r_frame12 = rotate_image(frame12,-5)
        
        H, C = panaroma.get_video_H_and_best_seam([r_frame12,frame34],mask,lowe_ratio=0.5)
        H_list.append(H), C_list.append(C)
        frame1234 = panaroma.image_stitch_for_video([r_frame12,frame34],H,C)
        
        init_flag = False

        # cv2.imwrite("results_r.jpg",frame1234[170:545,30:2100,:])

    else:
        H, C = H_list[0], C_list[0]
        frame34 = panaroma.image_stitch_for_video([frame3,frame4],H,C)
        
        H, C = H_list[1], C_list[1]
        frame12 = panaroma.image_stitch_for_video([frame1,frame2],H,C)
        r_frame12 = rotate_image(frame12,-5)

        H, C = H_list[2], C_list[2]        
        frame1234 = panaroma.image_stitch_for_video([r_frame12,frame34],H,C)
        # cv2.imwrite("results_r.jpg",frame1234[170:545,30:2100,:])
    
    end = time.time()
    fps = 1 / (end - start)

    # 定义文本信息
    text = "FPS = {}, Time = {}".format(fps, frame_cnt * 1 / fps)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.0
    org = (50, 250)
    color = (0, 255, 0)  # 绿色
    image = frame1234[170:545,30:2100,:] # 截取部分区域作为展示，消除黑色区域， 
    # 绘制文本
    cv2.putText(image, text, (700, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("panaroma",image)
    
    # 以30帧为速率进行播放
    cv2.waitKey(33)
    frame_cnt += 1

cap1.release()
cap2.release()
cap3.release()
cap4.release()
cv2.destroyAllWindows()



    