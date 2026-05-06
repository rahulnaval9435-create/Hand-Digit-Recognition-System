import cv2
import mediapipe as mp

# 1. SETUP FOR TWO HANDS
mp_hands = mp.solutions.hands
# Yahan 'max_num_hands=2' kiya hai taaki dono haath detect ho sakein
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# 2. CAMERA START
cap = cv2.VideoCapture(0)

print("Dono haath dikhayein! Band karne ke liye 'q' dabayein.")

try:
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        total_fingers = 0

        # Agar haath milte hain
        if results.multi_hand_landmarks:
            # Yeh loop har ek haath (Hand 1 aur Hand 2) ke liye chalega
            for hand_lms in results.multi_hand_landmarks:
                # Joints draw karein
                mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
                
                points = hand_lms.landmark
                fingers = []

                # Thumb Logic (Simple X-axis check)
                if points[4].x > points[3].x:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # 4 Fingers Logic (Y-axis check)
                tips = [8, 12, 16, 20]
                for tip in tips:
                    if points[tip].y < points[tip - 2].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # Is wale haath ki ungliyon ko total mein jod dein
                total_fingers += fingers.count(1)

        # 3. OUTPUT DISPLAY
        # Ek bada box aur total count
        cv2.rectangle(img, (20, 20), (400, 120), (0, 0, 0), -1)
        cv2.putText(img, f"Total Digits: {total_fingers}", (40, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

        cv2.imshow("Two Hand Recognition", img)

        # Exit logic
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Camera aur window ko puri tarah band karne ke liye
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    print("Program closed successfully.")