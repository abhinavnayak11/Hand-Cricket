import cv2
import numpy as np
import random
import os
import time

import torch
import torch.nn as nn
import torchvision

import albumentations as A


# get the model

model = torchvision.models.densenet121(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 500),
    # when using CrossEntropyLoss(), the output from NN should be logit values for each class
    # nn.Linear(500,1)
    # when using BCEWithLogitsLoss(), the output from NN should be logit value for True label
    nn.Linear(500, 7)
)

model.load_state_dict(torch.load('models/hand-cricket-model2.pth', map_location=torch.device('cpu')))
model.eval() 

# get the transform

augmentations = A.Compose([
    A.Resize(128, 128),
    A.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225), max_pixel_value = 255.0, always_apply=True
)
])


cap = cv2.VideoCapture(1)

i = 0
tf = 0

play_turn = 0
start_time = 0
game_started = 0
game_ended = 0
toss_done = 0 

count = 0
track_count = 0
player = None
human_score = 0
computer_score = 0

lower_white = np.array([230,230,230])
upper_white = np.array([255,255,255])

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)

    directory = "input"

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

    cv2.putText(frame, f"Score : {human_score}", (10, 30), font, fontscale, (0,255,255), thickness)
    cv2.putText(frame, f"Press 'space' to play", (460, 20), font, 1, (255,0,0), 2)
    cv2.putText(frame, f"Press 'r' to restart", (460, 40), font, 1, (255,0,0), 2)

    if game_started == 0:
        n_img_cut = frame[229:479,1:201]
        cv2.putText(frame, f"Toss", (250, 120), 0, 1.5, (0,255,255), 2)
        if toss_done == 0:
            
            if (interrupt & 0xFF == 104):   # press h for heads
                toss_done = 1
                toss = random.randint(0,1)  # 0 is heads, 1 is tails
                if toss == 0:    
                    player =  'human'
                else:
                    player = 'computer'
            if (interrupt & 0xFF == 116):   # press h for heads
                toss_done = 1
                toss = random.randint(0,1)  # 0 is heads, 1 is tails
                if toss == 1:    
                    player =  'human'
                else:
                    player = 'computer'
        if player:
            cv2.putText(frame, f"Player is {player}. Press space to play", (80, 160), font, 1.5, (0,255,255), 2)



    if (interrupt & 0xFF == 32) & (game_ended == 0): # press space to start the game
        game_started = 1
        play_turn = 1
        start_time = time.time()
        pred = torch.tensor(-1)

    if (game_started == 1) & (play_turn == 0):
        image_copy = np.copy(n_img)
        mask = cv2.inRange(image_copy, lower_white, upper_white)
        masked_image = np.copy(n_img)
        masked_image[mask != 0] = [0, 0, 0]
        crop_background = np.copy(frame[229:479,1:201])
        crop_background[mask == 0] = [0, 0, 0]
        n_img_cut = masked_image + crop_background
    
    elapsed = time.time() - start_time

    if (play_turn == 1) & (elapsed > 1):
        
        count+=1

        # computer play
        n = random.randint(0,6)
        n_img = cv2.imread(f'static/{n}.jpg')
        n_img = cv2.resize(n_img, (200,250))

        # removing background from computer images
        image_copy = np.copy(n_img)
        mask = cv2.inRange(image_copy, lower_white, upper_white)
        masked_image = np.copy(n_img)
        masked_image[mask != 0] = [0, 0, 0]
        crop_background = np.copy(frame[229:479,1:201])
        crop_background[mask == 0] = [0, 0, 0]
        n_img_cut = masked_image + crop_background

        # human play
        augmented = augmentations(image=roi)
        img = augmented["image"]
        img = img.transpose(2,0,1).astype(np.float32)
        img_tensor = torch.tensor(img)
        output = model(img_tensor.unsqueeze(0))
        pred = torch.argmax(output, dim=1)

        play_turn = 0

    if (game_started == 1) & (elapsed > 1):

        frame[229:479,1:201] = n_img_cut
        cv2.putText(frame, f"{pred.item()}", (539, 220), font, fontscale, (0,0,0), thickness)
        cv2.putText(frame, f"{n}", (100, 220), font, fontscale, (0,0,0), thickness)
    
    if count>track_count:
        if pred != n:
            human_score+=pred.item()
            track_count = count
        else:
            game_ended = 1
            count = 0
            track_count = 0
    
    if game_ended == 1:
        cv2.putText(frame, f" Game Over ", (120, 150), 0, fontscale, (0,255,255), thickness)
            

    if (interrupt & 0xFF == 114):  # restart game by pressing 'r'
        human_score = 0
        game_ended = 0
        game_started = 0
        toss_done = 0
        player = None



    cv2.imshow("Frame", frame)




cap.release()
cv2.destroyAllWindows()
