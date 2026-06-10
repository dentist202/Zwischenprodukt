import cv2                        #webcam + drawing on the image
import mediapipe as mp            #pose detection
import numpy as np                #math: vector, angle calculation
import matplotlib.pyplot as plt   #3D skeleton
from mpl_toolkits.mplot3d import Axes3D   #enables 3D plotting
 
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
 
 
# ============================================================
#                          CONSTANTS
# ============================================================


# 33 body points:

# 0 - nose
# 11 - left shoulder
# 12 - right shoulder
# 13 - left elbow
# 14 - right elbow
# 15 - left wrist
# 16 - right wrist
# 23 - left hip
# 24 - right hip
# 25 - left knee
# 26 - right knee
# 27 - left ankle
# 28 - right ankle
 
RIGHT_SHOULDER = 12
RIGHT_HIP = 24
RIGHT_KNEE = 26
RIGHT_ANKLE = 28
BASE_TORSO = 16.0  # виміряй при рівній стійці

# Pairs of points for SKELETONE:
POSE_CONNECTIONS = [
    (11, 12),                       # shoulders (left shoulder + right shoulder)
    (11, 13), (13, 15),             # left arm  (left shoulder + left elbow + left wrist)
    (12, 14), (14, 16),             # right arm (the same for right)
    (11, 23), (12, 24), (23, 24),   # torso     ((left shoulder + left hip) + (right shoulder + right hip) + (left hip + right hip))
    (23, 25), (25, 27),             # left leg  (left hip + left knee + left ankle)
    (24, 26), (26, 28),             # right leg (the same for right)
]
 

# ============================================================
#  CAMERA
# ============================================================



# Open the webcam (0 = default camera)
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("C:/Kanti3MATURA/matura-project/video/video5.mov")

 

# ============================================================
#  SETUP: model(pose_landmarker.task),
# ============================================================
 
 

# Mode_Setup (Video). 
# Video mode tracks landmarks across frames with timestamps (points are more predictible)
base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
options = PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=mp.tasks.vision.RunningMode.VIDEO, #VIDEO instead of IMAGE
)
pose_landmarker = PoseLandmarker.create_from_options(options)
 



# ============================================================
#  OPENCV
# ============================================================


cv2.resizeWindow("Pose Detection", cv2.WINDOW_NORMAL)   
cv2.setWindowProperty("Pose Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) 
fps = cap.get(cv2.CAP_PROP_FPS)


# 3D window for the skeleto
plt.ion()                                 
fig = plt.figure()                         # an empty figure without Axes
ax = fig.add_subplot(111, projection="3d")
 
 
# ============================================================
#                        FUNCTIONS
# ============================================================
 
def point_above(point, distance):
    # only y coordinate changes
    # media pipe Y is negativ, so we ahve to - distance
    return [point[0], point[1] - distance, point[2]]

def calculate_angle(point_a, point_b, point_c):

    point_a = np.array(point_a)
    point_b = np.array(point_b)
    point_c = np.array(point_c)
 
    vector_ba = point_a - point_b
    vector_bc = point_c - point_b
 
    #skalarprodukt
    cosine_angle = np.dot(vector_ba, vector_bc) / (
        np.linalg.norm(vector_ba) * np.linalg.norm(vector_bc)
    )


    # cos can be only between -1 nd 1
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
 
    return np.degrees(np.arccos(cosine_angle))
 
# ============================================================
#                        Landmarks
# ============================================================

#store landmarks in a dict {index:[x,y,z]}
def landmarks_to_dict(world_landmarks):

    positions = {}

    for index, landmark in enumerate(world_landmarks):

        positions[index] = [landmark.x, landmark.y, landmark.z]
    return positions
 


#to draw 2d skeleton on the screen (landmark(0-1)*screen_resolution)
def draw_skeleton_2d(frame, landmarks):
    for point_1, point_2 in POSE_CONNECTIONS:

        start = (int(landmarks[point_1].x * frame.shape[1]),
                 int(landmarks[point_1].y * frame.shape[0]))
        
        end = (int(landmarks[point_2].x * frame.shape[1]),
               int(landmarks[point_2].y * frame.shape[0]))
        cv2.line(frame, start, end, (41, 35, 35), 4)
 

# ============================================================
#                        matplotlib 3D Graph
# ============================================================
 

#to draw 3d skeleton on the grapth(matplotlib)
def draw_skeleton_3d(ax, world_landmarks):

    ax.cla() #to clear previous frame

    for point_1, point_2 in POSE_CONNECTIONS:
        ax.plot(
            [world_landmarks[point_1].x, world_landmarks[point_2].x],
            [world_landmarks[point_1].z, world_landmarks[point_2].z],
            [-world_landmarks[point_1].y, -world_landmarks[point_2].y],
            color="black", lw=2,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=10, azim=-90)
    plt.pause(0.001)    
 
 
# ============================================================
#                             MAIN LOOP
# ============================================================
 
def main():
    frame_index = 0    #count frames() timestamp
    graph_frame = 0  
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error")
            break
 
        #OpenCV(BGR) convert to MediaPipe(RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
 
        
        frame_index += 1
        timestamp_ms = int(frame_index * (1000 / fps))  #aprox 30 fps
        results = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
 
        #2D: draw skeleton on the screen + landmarks on the scrren
        if results.pose_landmarks:
            landmarks_2d = results.pose_landmarks[0]
            draw_skeleton_2d(frame, landmarks_2d)
 
            for landmark in landmarks_2d:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
 

        #3D: draw skeleton on the graph + right knee angle
        if results.pose_world_landmarks:
            world_landmarks = results.pose_world_landmarks[0]
            landmark_positions = landmarks_to_dict(world_landmarks)
 
            graph_frame += 1                        
            n = 1 # to change in what frequensy should points be displayed
            if graph_frame % n == 0:                
                draw_skeleton_3d(ax, world_landmarks)


            knee_angle = calculate_angle(
                landmark_positions[RIGHT_HIP],
                landmark_positions[RIGHT_KNEE],
                landmark_positions[RIGHT_ANKLE],
            )

            
            vertical_point = point_above(landmark_positions[RIGHT_HIP], distance = 0.5)

            torso_angle = calculate_angle(
                landmark_positions[RIGHT_SHOULDER],
                landmark_positions[RIGHT_HIP],                                  
                vertical_point, 
            )
            
            
            cv2.putText(frame, f"Right Knee Angle: {knee_angle:.1f}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Right Torso Lean Angle: {torso_angle:.1f}", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Pose Detection", frame)

        # "Q" to quit
        if cv2.waitKey(1) & 0xFF == ord("Q"):
            break
 
    cap.release()
    cv2.destroyAllWindows()
 
 
main()
 