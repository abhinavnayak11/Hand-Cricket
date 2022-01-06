import cv2
import numpy as np
import random
import os
import time

import torch
import torch.nn as nn
import torchvision

import albumentations as A


# define the model
model = torchvision.models.densenet121(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 500),  
    nn.Linear(500, 7)
)

# load the saved model
model.load_state_dict(torch.load('models/hand-cricket-densenet121.pth', map_location=torch.device('cpu')))
model.eval() 

# get the transform
augmentations = A.Compose([
    A.Resize(128, 128),
    A.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225), max_pixel_value = 255.0, always_apply=True
)
])

# create video capture instance
cap = cv2.VideoCapture(1)

play_turn = 0      # player playing one ball. 1 when playing. 0 when done.
start_time = 0     # time between pressing space and generating a number
game_started = 0   # 0 if game in standby (just before starting of each turn). 1 if game being played
game_ended = 0     # 1 if game comes to an end. 0 otherwise
toss_done = 0      # 1 if toss completed
toss_start_time = 0
toss_started = 0
change_turn = 0    # 0: 1st player, 1: 2nd player
instructions_window = 1

count = 0          # counts the number of balls played
track_count = 0    # tracks the count 
player = None      # either 'human' or 'computer'

player_scores = {'human':0,'computer':0}

lower_white = np.array([230,230,230])   # to mask hand images from static/
upper_white = np.array([255,255,255])   

font =  1   #  cv2.FONT_HERSHEY_PLAIN
fontscale = 2
thickness = 3
blue = (89,61,20)   
red = (0,0,255)
yellow = (26,180,244)

def transparent_background(n_img, right_frame):
    image_copy = np.copy(n_img)
    mask = cv2.inRange(image_copy, lower_white, upper_white)
    masked_image = np.copy(n_img)
    masked_image[mask != 0] = [0, 0, 0]
    crop_background = np.copy(right_frame)
    crop_background[mask == 0] = [0, 0, 0]
    n_img_cut = masked_image + crop_background
    return n_img_cut

def random_number_image():
    n = random.randint(0,6)
    n_img = cv2.imread(f'static/{n}.jpg')   # get the hand image of the generated number
    n_img = cv2.resize(n_img, (200,250))
    return n, n_img

def get_hand_prediction(roi):
    augmented = augmentations(image=roi)
    img = augmented["image"]
    img = img.transpose(2,0,1).astype(np.float32)
    img_tensor = torch.tensor(img)
    output = model(img_tensor.unsqueeze(0))
    pred = torch.argmax(output, dim=1)
    return pred

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)       # Simulating mirror image
    frame[0:190,0:640] = yellow      # upper part background color 

    # q key to close window
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == ord('q'): 
        break

    # instructions panel
    instructions = np.full((265,460,3), yellow).astype(np.uint8)
    cv2.putText(instructions, f"IMPORTANT!!", (150, 40), font, 1.5, (0,0,255), 2)
    cv2.putText(instructions, f"1. During toss, press 'o' for odd and 'e'", (10, 80), 3, 0.6, (0,0,0), 1)
    cv2.putText(instructions, f"for even and then position your hand", (30, 110), 3, 0.6, (0,0,0), 1)
    cv2.putText(instructions, f"inside the right frame", (30, 140), 3, 0.6, (0,0,0), 1)
    cv2.putText(instructions, f"2. When playing, press space and then", (10, 180), 3, 0.6, (0,0,0), 1)
    cv2.putText(instructions, f"place your hand inside the right frame", (30, 210), 3, 0.6, (0,0,0), 1)
    cv2.putText(instructions, f"Press esc to close", (302, 260), 2, 0.5, red, 1)

    # esc key to close instructions window
    if interrupt & 0xFF == 27: 
        instructions_window = 0
        cv2.destroyWindow('Instructions')   

    # box for human
    roi = frame[229:479, 439:639].copy()  # capture hand image from this roi
    cv2.rectangle(frame, (439, 229), (639, 479), (0,0,0), 2)
    cv2.putText(frame, f"You", (520, 220), font, fontscale, (0,0,0), 2)
    cv2.rectangle(frame, (600, 230), (639, 270), (0,0,0), 2)

    # box for computer
    cv2.rectangle(frame, (1, 229), (201, 479), (0,0,0), 2)
    cv2.putText(frame, f"Computer", (30, 220), font, fontscale, (0,0,0), 2)
    cv2.rectangle(frame, (161, 230), (200, 270), (0,0,0), 2)

    # upper right corner instructions
    cv2.putText(frame, f"Press 'space' to play", (460, 20), font, 1, red, 2)
    cv2.putText(frame, f"Press 'r' to restart", (460, 40), font, 1, red, 2)
    cv2.putText(frame, f"Press 'q' to quit", (460, 60), font, 1, red, 2)

    # To conduct toss when game starts
    if (game_started == 0) & (change_turn!=1):   
        if toss_done == 0:
            n_img_cut = frame[229:479,1:201]
        cv2.putText(frame, f"Toss", (280, 90), 0, 1.5, blue, 2)
        
        if toss_done == 0:
            
            cv2.putText(frame, f"Press 'o' for odd or 'e' for even", (100, 130), font, 1.5, blue, 2)

            if ((interrupt & 0xFF == 111) or (interrupt & 0xFF == 101)):   # press o for odd or e for even
                toss_started = 1
                if (interrupt & 0xFF == 111):
                    chosen = 1                  # 1 is odd
                else:
                    chosen = 0                  # 0 is even
                toss_start_time = time.time()
            
            toss_elapsed_time = time.time() - toss_start_time

            if ((toss_started == 1)&(toss_elapsed_time > 1)):

                toss_done = 1
                n, n_img = random_number_image()  
                pred = get_hand_prediction(roi)

                odd_or_even = (pred.item()+n)%2        # 0 is even, 1 is odd
                win = (odd_or_even == chosen)

                if win:    
                    player =  'human'
                else:
                    player = 'computer'
                
                toss = {0:'even', 1:'odd'}[odd_or_even]

        if player:
            frame[229:479,1:201] = transparent_background(n_img, frame[229:479,1:201])  # replacing left frame with number image (tranparent background)
            cv2.putText(frame, f"{pred.item()}", (610, 260), font, fontscale, (0,0,0), thickness)
            cv2.putText(frame, f"{n}", (170, 260), font, fontscale, (0,0,0), thickness)
            if player == 'human':
                cv2.putText(frame, f"It's {toss}!! You have won!", (150, 130), font, 1.5, blue, 2)
                cv2.putText(frame, f"You are the batsman. Press space to bat", (60, 160), font, 1.5, blue, 2)
            else:
                cv2.putText(frame, f"It's {toss}!! You have lost!", (150, 130), font, 1.5, blue, 2)
                cv2.putText(frame, f"Computer is the batsman. Press space to bowl", (20, 160), font, 1.5, blue, 2)
            

    # display the score and who is the batsman/bowler
    if ((game_started == 1) or ((change_turn == 1) & (game_started == 0))):
        if player == 'human':
            cv2.putText(frame, f"Your Score (Batsman) : {player_scores['human']}", (10, 20), font, 1.2, blue, 2)
            cv2.putText(frame, f"Computer Score (Bowler) : {player_scores['computer']}", (10, 40), font, 1.2, blue, 2)
        if player == 'computer':
            cv2.putText(frame, f"Your Score (Bowler) : {player_scores['human']}", (10, 20), font, 1.2, blue, 2)
            cv2.putText(frame, f"Computer Score (Batsman) : {player_scores['computer']}", (10, 40), font, 1.2, blue, 2)

    # when turn changes, change the player
    if change_turn == 1:
        if player == 'human':
            first_player = 'computer'
        else:
            first_player = 'human'
        if player_scores[player] > player_scores[first_player]:
            game_ended = 1   # when second player score more than first game ends

    # when player 1 gets out and turn changes
    if (change_turn == 1) & (game_started == 0):
        cv2.putText(frame, f"That's OUT!!!", (230, 120), font, 2, blue, 2)
        if player == 'human':
            cv2.putText(frame, f"Now, you are the batsman. You need {player_scores[first_player]+1} runs to win", (50, 160), font, 1.2, blue, 2)
        else:
            cv2.putText(frame, f"Now, computer is the batsman and needs {player_scores[first_player]+1} runs to win", (30, 160), font, 1.2, blue, 2)

    # press space to start the game. 
    # won't work if toss not done yet
    # won't work if game has ended. Need to press 'r' to restart
    if (interrupt & 0xFF == 32) & (game_ended == 0) & (toss_done == 1): 
        game_started = 1
        play_turn = 1
        start_time = time.time()
        pred = torch.tensor(-1)

    # to mask the hand images from static folder
    if (((game_started == 1) & (play_turn == 0)) or ((change_turn == 1) & (game_started == 0))):
        n_img_cut = transparent_background(n_img, frame[229:479,1:201])
    
    # get the time elapsed after pressing space
    elapsed = time.time() - start_time

    # play one ball after 0.5 second of pressing space
    if (play_turn == 1) & (elapsed > 1):
        
        count+=1

        # computer play
        n, n_img = random_number_image()

        # removing white background from computer images
        n_img_cut = transparent_background(n_img, frame[229:479,1:201])

        # human play
        pred = get_hand_prediction(roi)

        play_turn = 0

    # display each players number after each ball 
    if (((game_started == 1) & (elapsed > 1)) or ((change_turn == 1) & (game_started == 0))):

        frame[229:479,1:201] = n_img_cut   # replacing left frame with number image (tranparent background)
        cv2.putText(frame, f"{pred.item()}", (610, 260), font, fontscale, (0,0,0), thickness)
        cv2.putText(frame, f"{n}", (170, 260), font, fontscale, (0,0,0), thickness)
    
    # count the score after each ball 
    if count>track_count:
        # when different numbers played, add the player's number to score
        if pred.item() != n:
            if player == 'human':
                if pred.item() == 0:         # add opposite player's number when 0 is played (that's how we used to play :p)
                    player_scores[player]+=n  
                else:
                    player_scores[player]+=pred.item()
            else:
                if n == 0:                   # add opposite player's number when 0 is played
                    player_scores[player]+=pred.item()
                else:
                    player_scores[player]+=n
                
            track_count = count

        # when 1st player gets out, change turn
        elif (pred.item() == n) & (change_turn == 0):
            change_turn = 1
            if player == 'human':
                player = 'computer'
            else:
                player = 'human'
            count = 0
            track_count = 0
            game_started = 0

        # when both players have finished playing, end the game
        else:
            game_ended = 1
            count = 0
            track_count = 0
            change_turn = 0
    
    # display results after game ends
    if game_ended == 1:
        cv2.putText(frame, f" Game Over ", (170, 120), 0, 1.5, blue, thickness)
        if player_scores['human'] > player_scores['computer']:
            cv2.putText(frame, f"You win", (250, 170), font, fontscale, blue, 2)
        elif player_scores['human'] < player_scores['computer']:
            cv2.putText(frame, f"Computer wins", (200, 170), font, fontscale, blue, 2)
        else:
            cv2.putText(frame, f"It's a draw", (230, 170), font, fontscale, blue, 2)
            
    # restart the game when 'r' is pressed
    if (interrupt & 0xFF == 114):  
        player_scores = {'human':0,'computer':0}
        game_ended = 0
        game_started = 0
        toss_done = 0
        toss_started = 0
        player = None
        change_turn = 0

    cv2.rectangle(frame, (0, 190), (640, 190), (0,0,0), 2)   # line at the end of instructions panel
    cv2.rectangle(frame, (0, 0), (640, 480), (0,0,0), 2)     # border around the whole frame

    cv2.imshow("Hand Cricket", frame)

    if instructions_window == 1:
        cv2.imshow("Instructions", instructions)

cap.release()
cv2.destroyAllWindows()