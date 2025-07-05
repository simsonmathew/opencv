import cv2
import mediapipe as mp  # type: ignore
import time
import pygame  # type: ignore
import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

pygame.mixer.init()
pygame.mixer.music.set_volume(1.0) 

try:
    pygame.mixer.music.load("die with a smile.mp3.mp3")
except pygame.error as e:
    print(f"Error loading audio file: {e}")
    exit()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

GESTURE_ACTIONS = {
    "thumbs_up": "play_music",
    "fist": "stop_music",
    "open_palm": "pause_music"
}

def recognize_gesture(landmarks):
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]

    fingers = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]

    if all(abs(f.y - wrist.y) < 0.05 for f in fingers):
        return "fist"

    if thumb_tip.y < index_tip.y and \
       all(f.y > wrist.y for f in [index_tip, middle_tip, ring_tip, pinky_tip]) and \
       abs(thumb_tip.x - index_tip.x) > 0.1:
        return "thumbs_up"

    finger_above_wrist = all(f.y < wrist.y - 0.05 for f in fingers)
    x_coords = [f.x for f in fingers]
    spread = max(x_coords) - min(x_coords)
    if finger_above_wrist and spread > 0.3:
        return "open_palm"

    return None

def control_music(action):
    try:
        if action == "play_music":
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play(-1) 
                print("Music Playing")
        elif action == "pause_music":
            pygame.mixer.music.pause()
            print("Music Paused")
        elif action == "stop_music":
            pygame.mixer.music.stop()
            print("Music Stopped")
    except pygame.error as e:
        print(f"Music control error: {e}")

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    last_gesture_time = time.time()
    last_gesture = None
    debounce_seconds = 2

    print("Gesture Music Control Running... (Press ESC to Exit)")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = hands.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                gesture = recognize_gesture(hand_landmarks)
                now = time.time()

                if gesture and gesture != last_gesture and (now - last_gesture_time) > debounce_seconds:
                    action = GESTURE_ACTIONS.get(gesture)
                    if action:
                        control_music(action)
                        last_gesture = gesture
                        last_gesture_time = now
        else:
            last_gesture = None

        cv2.imshow("Gesture Control", frame)
        if cv2.waitKey(5) & 0xFF == 27:  
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    pygame.mixer.quit()
main()
