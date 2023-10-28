import numpy as np
import time
import cv2
import mediapipe as mp
import time
import urllib

mp_hands = mp.solutions.hands

CAPTURE_URL = 'http://192.168.88.17/capture'
FINTER_TIP_IDS = [
    mp_hands.HandLandmark.THUMB_TIP,
    mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp_hands.HandLandmark.RING_FINGER_TIP,
    mp_hands.HandLandmark.PINKY_TIP,
]

last_time = time.time()

def calc_angle_radians(vector_a, vector_b) -> float:
    dot_product = np.dot(vector_a, vector_b)
    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)
    angle_radians = np.arccos(dot_product / (magnitude_a * magnitude_b))
    return angle_radians


def is_finder_straight(hand_landmarks, tip_id) -> bool:
    def make_point(id):
        landmark = hand_landmarks.landmark[id]
        return np.array([landmark.x, landmark.y, landmark.z])

    vectors = [
        make_point(mp_hands.HandLandmark.WRIST) - make_point(tip_id - 3),
        make_point(tip_id - 3) - make_point(tip_id - 2),
        make_point(tip_id - 2) - make_point(tip_id - 1),
        make_point(tip_id - 1) - make_point(tip_id),
    ]

    angles_radians = [calc_angle_radians(vectors[i], vectors[i+1]) for i in range(len(vectors) - 1)]
    # print('finder', tip_id, angles_radians)

    return angles_radians[0] < np.pi/2 and angles_radians[1] < np.pi/6 and angles_radians[2] < np.pi/6


if not CAPTURE_URL:
    print("Inializing video capture...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Done")

# https://github.com/google/mediapipe/blob/master/docs/solutions/hands.md
print("Inializing hands model...")
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
)
print("Done")

if __name__ == "__main__":
    while True:
        if CAPTURE_URL:
            img_resp = urllib.request.urlopen(CAPTURE_URL)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            img = cv2.imdecode(imgnp, -1)
            img_rgb = img
        else:
            success, img = cap.read()
            if not success:
                print("Capturing failed")
                time.sleep(1)
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        finder_straight = []
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp.solutions.drawing_utils.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            finder_straight = [is_finder_straight(hand_landmarks, tip_id) for tip_id in FINTER_TIP_IDS]

        now = time.time()
        fps = int(1 / (now - last_time))
        last_time = now

        cv2.putText(img=img,
                    text=f'FPS {fps}',
                    org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 255),
                    thickness= 1)
        cv2.putText(img=img,
                    text=f'{finder_straight.count(True)}: {finder_straight}',
                    org=(10, 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 255),
                    thickness= 1)

        cv2.imshow("Fingers Count", img)

        escape_key_code = 27
        if cv2.waitKey(1) & 0xFF == escape_key_code:
            break

    cv2.destroyAllWindows()