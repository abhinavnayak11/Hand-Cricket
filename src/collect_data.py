# Credits: 
# - Adarsh Menon : https://github.com/chasinginfinity/number-sign-recognition/blob/master/collect-data.py 

import cv2
import numpy as np
import os


cap = cv2.VideoCapture(1)

i = 0

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)

    directory = "input"

    count = {'zero': len(os.listdir(directory+"/0")),
             'one': len(os.listdir(directory+"/1")),
             'two': len(os.listdir(directory+"/2")),
             'three': len(os.listdir(directory+"/3")),
             'four': len(os.listdir(directory+"/4")),
             'five': len(os.listdir(directory+"/5")),
             'six':len(os.listdir(directory+"/6"))}

    cv2.putText(frame, "IMAGE COUNT", (10, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)

    font =  1   #  cv2.FONT_HERSHEY_PLAIN
    fontscale = 1.5
    thickness = 1
    
    cv2.putText(frame, f"Zero :   {count['zero']}", (10, 20), font, fontscale, (0,255,255), thickness)
    cv2.putText(frame, f"One :    {count['one']}", (10, 40), font, fontscale, (0,255,255), thickness)
    cv2.putText(frame, f"Two :    {count['two']}", (10, 60), font, fontscale, (0,255,255), thickness)
    cv2.putText(frame, f"Three :  {count['three']}", (10, 80), font, fontscale, (0,255,255), thickness)
    cv2.putText(frame, f"Four :   {count['four']}", (10, 100), font, fontscale, (0,255,255), thickness)
    cv2.putText(frame, f"Five :    {count['five']}", (10, 120), font, fontscale, (0,255,255), thickness)
    cv2.putText(frame, f"Six :     {count['six']}", (10, 140), font, fontscale, (0,255,255), thickness)

    box = 3
    
    if box == 1:
        roi = frame[1:251, 439:639].copy()   # To ensure rectangle drawn on 'frame' is not reflected in 'roi'
        cv2.rectangle(frame, (439,1), (639, 251), (0,0,0), 2)

    if box == 2:
        roi = frame[90:340, 260:460].copy()
        cv2.rectangle(frame, (260, 90), (460, 340), (0,0,0), 2)

    if box == 3:
        roi = frame[144:394, 439:639].copy()
        cv2.rectangle(frame, (439,144), (639, 394), (0,0,0), 2)
 
    cv2.imshow("Frame", frame)
    cv2.imshow('roi', roi)
    

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break

    if interrupt & 0xFF == ord('0'):
        cv2.imwrite(f"input/0/zero{count['zero']}.jpg", roi)
    if interrupt & 0xFF == ord('1'):
        cv2.imwrite(f"input/1/one{count['one']}.jpg", roi)
    if interrupt & 0xFF == ord('2'):
        cv2.imwrite(f"input/2/two{count['two']}.jpg", roi)
    if interrupt & 0xFF == ord('3'):
        cv2.imwrite(f"input/3/three{count['three']}.jpg", roi)
    if interrupt & 0xFF == ord('4'):
        cv2.imwrite(f"input/4/four{count['four']}.jpg", roi)
    if interrupt & 0xFF == ord('5'):
        cv2.imwrite(f"input/5/five{count['five']}.jpg", roi)
    if interrupt & 0xFF == ord('6'):
        cv2.imwrite(f"input/6/six{count['six']}.jpg", roi)


cap.release()
cv2.destroyAllWindows()