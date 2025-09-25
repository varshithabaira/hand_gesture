import cv2
import numpy as np
import pandas as pd
import pickle
import mediapipe as mp
from multiprocessing import Process, Manager
import time
import os
import warnings
warnings.filterwarnings('ignore')


with open("hand_sign_model.pkl", "rb") as h:
    model = pickle.load(h)

with open("label_encoder.pkl", "rb") as h:
    label_encoder = pickle.load(h)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  

def extract_landmarks(results):
    if not results.multi_hand_landmarks:
        return None
    hand_landmarks = results.multi_hand_landmarks[0]
    landmarks = [] 
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return np.array(landmarks).reshape(1, -1)

def detect_hand(cam_id, player_name, filename, result_holder):
    cap = cv2.VideoCapture(cam_id)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.8, min_tracking_confidence=0.8)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        label = result_holder.get('result', "") 
        if results.multi_hand_landmarks:        
            for hand_landmarks in results.multi_hand_landmarks:   
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            features = extract_landmarks(results)
            if features is not None:
                df = pd.DataFrame(features, columns=[str(i) for i in range(features.shape[1])])
                pred = model.predict(df)
                
                label = label_encoder.inverse_transform(pred)[0]  
                with open(filename, "w") as h:  
                    h.write(label) 
        if label:
            cv2.putText(frame, f"{player_name}: {label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        if result_holder.get('display', ""):
            cv2.putText(frame, result_holder['display'], (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

        cv2.imshow(f"{player_name} - Camera {cam_id}", frame) #
        if cv2.waitKey(1) & 0xFF == ord('q'):
            with open(filename, "w") as f:
                f.write("quit")
            break

    cap.release()
    cv2.destroyAllWindows()




def game_manager(p1_name, p2_name, result_holder1, result_holder2):
    win_map = {"rock": "scissors", "scissors": "paper", "paper": "rock"}
    scores = {p1_name: 0, p2_name: 0}


    while True:
        try:
            with open("player1.txt") as f1, open("player2.txt") as f2:
                move1 = f1.read().strip()
                move2 = f2.read().strip()

            if move1 == "quit" or move2 == "quit":
                break

            if move1 and move2:
                if move1 == move2:
                    result_holder1['display'] = "Result: Tie!"
                    result_holder2['display'] = "Result: Tie!"
                elif win_map.get(move1) == move2:
                    scores[p1_name] += 1                                                        
                    scores[p2_name] = max(0, scores[p2_name] - 1)     
                    result_holder1['display'] = f"{p1_name} Wins! ({scores[p1_name]})" 
                    result_holder2['display'] = f"{p1_name} Wins!" 
                else:
                    scores[p2_name] += 1     
                    scores[p1_name] = max(0, scores[p1_name] - 1)
                    result_holder1['display'] = f"{p2_name} Wins!"  
                    result_holder2['display'] = f"{p2_name} Wins! ({scores[p2_name]})" 

                time.sleep(2)


                with open("player1.txt", "w") as f1, open("player2.txt", "w") as f2:
                    f1.write("")
                    f2.write("")
        except FileNotFoundError:
            time.sleep(0.5)

if __name__ == '__main__':
    player1_name = input("Enter Player 1 name: ")
    player2_name = input("Enter Player 2 name: ")


    open("player1.txt", "w").close()
    open("player2.txt", "w").close()
    

    with Manager() as manager:
        result_holder1 = manager.dict()
        result_holder2 = manager.dict()

        p1 = Process(target=detect_hand, args=(0, player1_name, "player1.txt", result_holder1))
        p2 = Process(target=detect_hand, args=(1, player2_name, "player2.txt", result_holder2))

        p1.start()
        p2.start()

        p1.join()
        p2.join()














    os.remove("player1.txt")
    os.remove("player2.txt")