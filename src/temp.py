import cv2
import numpy as np
import random


cap = cv2.VideoCapture(1)


lower_white = np.array([230,230,230])
upper_white = np.array([255,255,255])

blue = (89,61,20)
red = (0,0,255)
yellow = (0,255,255)

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    frame[0:190,0:640] = (26,180,244)

    font =  1   #  cv2.FONT_HERSHEY_PLAIN
    fontscale = 2
    thickness = 3

    # get the hand images from this roi
    roi = frame[229:479, 439:639].copy()
    cv2.rectangle(frame, (439, 229), (639, 479), (0,0,0), 2)
    

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break


    cv2.rectangle(frame, (1, 229), (201, 479), (0,0,0), 2)

    cv2.putText(frame, f"Press 'space' to play", (460, 20), font, 1, red, 2)
    cv2.putText(frame, f"Press 'r' to restart", (460, 40), font, 1, red, 2)

    cv2.putText(frame, f"Toss", (280, 90), 0, 1.5, blue, 2)
    cv2.putText(frame, f"Press 'h' for heads or 't' for tails", (100, 130), font, 1.5, blue, 2)
    cv2.putText(frame, f"Heads it is!!!", (250, 130), font, 1.5, blue, 2)
    cv2.putText(frame, f"Batsman is Human. Press space to play", (60, 160), font, 1.5, blue, 2)

    cv2.putText(frame, f"Your Score (Batsman) : 34", (10, 20), font, 1.2, blue, 2)
    cv2.putText(frame, f"Computer Score (Bowler) : 44", (10, 40), font, 1.2, blue, 2)

    cv2.putText(frame, f"That's OUT!!!", (230, 120), font, 2, blue, 2) 


    cv2.putText(frame, f"Now, the batsman is Human. You need 33 runs to win", (50, 160), font, 1.2, blue, 2)
        

    cv2.putText(frame, f"{5}", (610, 260), font, fontscale, (0,0,0), thickness)
    cv2.putText(frame, f"{0}", (170, 260), font, fontscale, (0,0,0), thickness)



    cv2.putText(frame, f" Game Over ", (170, 120), 0, 1.5, blue, thickness)
    cv2.putText(frame, f"You win", (250, 170), font, fontscale, blue, 2)
    cv2.putText(frame, f"Computer wins", (200, 170), font, fontscale, blue, 2)
    cv2.putText(frame, f"It's a draw", (230, 170), font, fontscale, blue, 2)

    cv2.rectangle(frame, (0, 0), (640, 190), (0,0,0), 2)
    cv2.rectangle(frame, (161, 230), (200, 270), (0,0,0), 2)
    cv2.rectangle(frame, (600, 230), (639, 270), (0,0,0), 2)

    cv2.putText(frame, f"Computer", (30, 220), font, fontscale, (0,0,0), 2)
    cv2.putText(frame, f"You", (520, 220), font, fontscale, (0,0,0), 2)

    cv2.imshow("Frame", frame)

cap.release()
cv2.destroyAllWindows()
