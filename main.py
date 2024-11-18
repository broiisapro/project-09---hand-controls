import cv2
import mediapipe as mp
import math
import pyautogui
import screeninfo

# Initialize MediaPipe Hands and drawing utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Initialize the camera
camera = cv2.VideoCapture(0)

# Get screen size to map hand position to screen coordinates
screen = screeninfo.get_monitors()[0]
screen_width, screen_height = screen.width, screen.height

# Function to calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

# List of landmark indices for finger tips
FINGER_TIPS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

# Larger radius for finger tip landmarks
TIP_RADIUS = 9  # Increase this value for larger dots at finger tips

# Function to move the mouse based on hand position
def move_mouse(x, y):
    # Map the x, y coordinates of the hand to screen dimensions
    mouse_x = int(x * screen_width)
    mouse_y = int(y * screen_height)
    pyautogui.moveTo(mouse_x, mouse_y)

# Function to simulate left click
def left_click():
    pyautogui.click()

# Function to simulate right click
def right_click():
    pyautogui.rightClick()

# Function to scroll up
def scroll_up():
    pyautogui.scroll(10)

# Function to scroll down
def scroll_down():
    pyautogui.scroll(-10)

# Function to detect gestures and control the mouse
def detect_gesture(finger_coords):
    # Calculate the distance between thumb and index finger (for pinch gestures)
    thumb_index_dist = calculate_distance(finger_coords[0], finger_coords[1])
    
    # Calculate the distance between thumb and pinky (for open hand gestures)
    thumb_pinky_dist = calculate_distance(finger_coords[0], finger_coords[4])
    
    # Left click (pinch gesture)
    if thumb_index_dist < 50:  
        left_click()
    
    # Right click (two-finger pointing gesture)
    elif thumb_index_dist > 150 and calculate_distance(finger_coords[1], finger_coords[2]) < 50:  
        right_click()
    
    # Scroll up (thumb up gesture)
    elif finger_coords[4][1] < finger_coords[0][1]:  
        scroll_up()
    
    # Scroll down (thumb down gesture)
    elif finger_coords[4][1] > finger_coords[0][1]:  
        scroll_down()

# Main loop for hand tracking and mouse control
while True:
    # Capture frame-by-frame
    ret, image = camera.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Flip the image horizontally for a mirror effect
    image = cv2.flip(image, 1)

    # Convert the image to RGB (required by MediaPipe)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and get hand landmarks
    results = hands.process(rgb_image)

    # Check if any hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw connections between landmarks (white lines)
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Store coordinates for each finger tip
            finger_coords = []

            for tip_index in FINGER_TIPS:
                lm = hand_landmarks.landmark[tip_index]
                x, y = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
                finger_coords.append((x, y))
                # Draw blue circle at each finger tip with a black outline
                cv2.circle(image, (x, y), TIP_RADIUS, (0, 0, 0), 2)  # Black outline
                cv2.circle(image, (x, y), TIP_RADIUS - 2, (255, 0, 0), -1)  # Blue fill

            # Calculate the center of the hand (using wrist and finger points)
            wrist = hand_landmarks.landmark[0]  # Wrist is landmark 0
            hand_center = (wrist.x, wrist.y)  # Center is the wrist position for simplicity

            # Move the mouse based on the hand's position
            move_mouse(hand_center[0], hand_center[1])

            # Detect gestures and perform actions
            detect_gesture(finger_coords)

    # Display the image with landmarks and distances
    cv2.imshow("Hand Tracking for Mouse Control", image)

    # Exit the loop when the 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the camera and close OpenCV windows
camera.release()
cv2.destroyAllWindows()
