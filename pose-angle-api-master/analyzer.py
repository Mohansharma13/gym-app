import cv2
import mediapipe as mp
import math

class PoseAnalyzer:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose()
        self.min_elbow, self.max_elbow = 180, 0
        self.min_knee, self.max_knee = 180, 0

    def calculate_angle(self, a, b, c):
        ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - 
                           math.atan2(a[1] - b[1], a[0] - b[0]))
        ang = abs(ang)
        return ang if ang <= 180 else 360 - ang

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            if not results.pose_landmarks:
                continue

            h, w, _ = frame.shape
            lm = results.pose_landmarks.landmark

            # Arm (right): Shoulder - Elbow - Wrist
            a1 = (int(lm[12].x * w), int(lm[12].y * h))  # shoulder
            b1 = (int(lm[14].x * w), int(lm[14].y * h))  # elbow
            c1 = (int(lm[16].x * w), int(lm[16].y * h))  # wrist

            # Leg (right): Hip - Knee - Ankle
            a2 = (int(lm[24].x * w), int(lm[24].y * h))  # hip
            b2 = (int(lm[26].x * w), int(lm[26].y * h))  # knee
            c2 = (int(lm[28].x * w), int(lm[28].y * h))  # ankle

            elbow_angle = self.calculate_angle(a1, b1, c1)
            knee_angle = self.calculate_angle(a2, b2, c2)

            self.min_elbow = min(self.min_elbow, elbow_angle)
            self.max_elbow = max(self.max_elbow, elbow_angle)
            self.min_knee = min(self.min_knee, knee_angle)
            self.max_knee = max(self.max_knee, knee_angle)

        cap.release()
        return {
            "elbow_min_angle": round(self.min_elbow, 2),
            "elbow_max_angle": round(self.max_elbow, 2),
            "knee_min_angle": round(self.min_knee, 2),
            "knee_max_angle": round(self.max_knee, 2),
        }
