import cv2
import numpy as np
import pyautogui

# Set up OpenCV
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Define color ranges for hand detection (you may need to adjust these values)
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# Initialize variables
keyboard_layout = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/']
]
current_key = ''

while True:
    ret, frame = cap.read()

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the frame to get only the skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (hand)
    if contours:
        hand_contour = max(contours, key=cv2.contourArea)

        # Get the centroid of the hand
        M = cv2.moments(hand_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Check if the centroid is within a key region
            for row, keys in enumerate(keyboard_layout):
                for col, key in enumerate(keys):
                    x, y, w, h = col * 60, row * 60, 60, 60
                    if x < cx < x + w and y < cy < y + h:
                        current_key = key
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red color
                        break

    # Display the keyboard layout on the frame
    for row, keys in enumerate(keyboard_layout):
        for col, key in enumerate(keys):
            x, y = col * 60, row * 60
            cv2.putText(frame, key, (x + 20, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Red color

    # Display the current key being pressed
    cv2.putText(frame, f"Current Key: {current_key}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow('Virtual Keyboard', frame)

    # Break the loop if 'Esc' is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
