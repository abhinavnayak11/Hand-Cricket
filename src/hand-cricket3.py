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

random.seed(10)

play_turn = 0
start_time = 0
game_started = 0
game_ended = 0
toss_done = 0 
change_turn = 0    # 0: 1st player, 1: 2nd player

count = 0
track_count = 0
player = None

player_scores = {'human':0,'computer':0}

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

    # box for human
    roi = frame[229:479, 439:639].copy()
    cv2.rectangle(frame, (439, 229), (639, 479), (0,0,0), 2)
    cv2.putText(frame, f"You", (520, 220), font, fontscale, (0,0,0), 2)
    cv2.rectangle(frame, (600, 230), (639, 270), (0,0,0), 2)
    

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break

    # box for computer
    cv2.rectangle(frame, (1, 229), (201, 479), (0,0,0), 2)
    cv2.putText(frame, f"Computer", (30, 220), font, fontscale, (0,0,0), 2)
    cv2.rectangle(frame, (161, 230), (200, 270), (0,0,0), 2)

    cv2.putText(frame, f"Press 'space' to play", (460, 20), font, 1, red, 2)
    cv2.putText(frame, f"Press 'r' to restart", (460, 40), font, 1, red, 2)

    if (game_started == 0) & (change_turn!=1):
        n_img_cut = frame[229:479,1:201]
        cv2.putText(frame, f"Toss", (280, 90), 0, 1.5, blue, 2)
        if toss_done == 0:
            cv2.putText(frame, f"Press 'h' for heads or 't' for tails", (100, 130), font, 1.5, blue, 2)
            if (interrupt & 0xFF == 104):   # press h for heads
                toss_done = 1
                toss = random.randint(0,1)  # 0 is heads, 1 is tails
                if toss == 0:    
                    player =  'human'
                    toss = 'Heads'
                else:
                    player = 'computer'
                    toss = 'Tails'
            if (interrupt & 0xFF == 116):   # press h for heads
                toss_done = 1
                toss = random.randint(0,1)  # 0 is heads, 1 is tails
                if toss == 1:    
                    player =  'human'
                    toss = 'Tails'
                else:
                    player = 'computer'
                    toss = 'Heads'
        if player:
            cv2.putText(frame, f"{toss} it is!!!", (250, 130), font, 1.5, blue, 2)
            cv2.putText(frame, f"Batsman is {player}. Press space to play", (60, 160), font, 1.5, blue, 2)

    # if (game_started == 1) & (change_turn != 1):
    #     cv2.putText(frame, f"Score : {player_scores[player]}", (10, 30), font, fontscale, (0,255,255), thickness)
    # if change_turn == 1 or game_ended == 1:
    #     cv2.putText(frame, f"Your Score : {player_scores['human']}", (10, 30), font, fontscale, (0,255,255), thickness)
    #     cv2.putText(frame, f"Computer Score : {player_scores['computer']}", (10, 60), font, fontscale, (0,255,255), thickness)


    if ((game_started == 1) or ((change_turn == 1) & (game_started == 0))):
        if player == 'human':
            cv2.putText(frame, f"Your Score (Batsman) : {player_scores['human']}", (10, 20), font, 1.2, blue, 2)
            cv2.putText(frame, f"Computer Score (Bowler) : {player_scores['computer']}", (10, 40), font, 1.2, blue, 2)
        if player == 'computer':
            cv2.putText(frame, f"Your Score (Bowler) : {player_scores['human']}", (10, 20), font, 1.2, blue, 2)
            cv2.putText(frame, f"Computer Score (Batsman) : {player_scores['computer']}", (10, 40), font, 1.2, blue, 2)

    if (change_turn == 1) & (game_started == 0):
        cv2.putText(frame, f"That's OUT!!!", (230, 120), font, 2, blue, 2)

    if change_turn == 1:
        if player == 'human':
            first_player = 'computer'
        else:
            first_player = 'human'
        if player_scores[player] > player_scores[first_player]:
            game_ended = 1

    if (change_turn == 1) & (game_started == 0):
        cv2.putText(frame, f"Now, the batsman is {player}. You need {player_scores[first_player]} runs to win", (40, 160), font, 1.2, blue, 2)
            
    
    if (interrupt & 0xFF == 32) & (game_ended == 0): # press space to start the game
        game_started = 1
        play_turn = 1
        start_time = time.time()
        pred = torch.tensor(-1)

    if (((game_started == 1) & (play_turn == 0)) or ((change_turn == 1) & (game_started == 0))):
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

    if (((game_started == 1) & (elapsed > 1)) or ((change_turn == 1) & (game_started == 0))):

        frame[229:479,1:201] = n_img_cut
        cv2.putText(frame, f"{pred.item()}", (610, 260), font, fontscale, (0,0,0), thickness)
        cv2.putText(frame, f"{n}", (170, 260), font, fontscale, (0,0,0), thickness)
    
    
    if count>track_count:
        if pred.item() != n:
            if player == 'human':
                player_scores[player]+=pred.item()
            else:
                player_scores[player]+=n
            track_count = count

        elif (pred.item() == n) & (change_turn == 0):
            change_turn = 1
            if player == 'human':
                player = 'computer'
            else:
                player = 'human'
            count = 0
            track_count = 0
            game_started = 0

        else:
            game_ended = 1
            count = 0
            track_count = 0
            change_turn = 0
    
    if game_ended == 1:
        cv2.putText(frame, f" Game Over ", (170, 120), 0, 1.5, blue, thickness)
        if player_scores['human'] > player_scores['computer']:
            cv2.putText(frame, f"You win", (250, 170), font, fontscale, blue, 2)
        elif player_scores['human'] < player_scores['computer']:
            cv2.putText(frame, f"Computer wins", (200, 170), font, fontscale, blue, 2)
        else:
            cv2.putText(frame, f"It's a draw", (230, 170), font, fontscale, blue, 2)
            

    if (interrupt & 0xFF == 114):  # restart game by pressing 'r'
        player_scores = {'human':0,'computer':0}
        game_ended = 0
        game_started = 0
        toss_done = 0
        player = None
        change_turn = 0

    cv2.rectangle(frame, (0, 190), (640, 190), (0,0,0), 2)
    cv2.rectangle(frame, (0, 0), (640, 480), (0,0,0), 2)

    cv2.imshow("Odd or Even (The game)", frame)

cap.release()
cv2.destroyAllWindows()
