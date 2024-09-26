import cv2
import mediapipe as mp
import json
import socket
import pyrealsense2 as rs
import numpy as np
import statistics
import time

def send_data(data, host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    
    # Convert the list of lists to a string
    message = json.dumps(data)
    
    s.sendall(message.encode('utf-8'))
    
    # Close the connection
    s.close()



host = '192.168.0.130'  # Replace with the receiver's IP address
port = 10000  # The port you've chosen

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
left_hand = 0
right_hand = 0

pre_body_depth = 0
pre_left_depth = 0
pre_right_depth = 0
center_shoulder_x = 0.5
center_shoulder_y = 0.5
body_depth = 600
left_depth = 550
right_depth = 550

mean_body_depth = [0]
mean_left_depth = [0]
mean_right_depth = [0]

long_factor = 1.3
high_factor = 1
width_factor = 1

run_time = 0

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
print("device_product_line = ",device_product_line)

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)
high_factor = 1
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
send_delay_counter = 0
movement_list = [0.15, 0.2, 0.3, 0.15, -0.2, 0.3, 0.8]

while True:
    pre_time = time.time()
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not depth_frame or not color_frame:
            continue

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    rgb_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    depth_colormap_dim = depth_colormap.shape
    # print("depth_colormap_dim = ",depth_colormap_dim)
    # Process the frame and get pose landmarks
    result = pose.process(rgb_frame)

    # Draw the pose annotations on the frame
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            color_image,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )

        landmarks = result.pose_landmarks.landmark
        left_hand = landmarks[mp_pose.PoseLandmark.LEFT_INDEX]
        right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        
        center_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
        center_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

        print("Center = ", (left_shoulder.x + right_shoulder.x) / 2)

    if left_hand and right_hand and right_shoulder and left_shoulder:

        if center_shoulder_x > 0.99:
            center_shoulder_x = 0.99
        if center_shoulder_y > 0.99:
            center_shoulder_y = 0.99
        if left_hand.x > 0.99:
            left_hand.x = 0.99
        if left_hand.y > 0.99:
            left_hand.y = 0.99
        if right_hand.x > 0.99:
            right_hand.x = 0.99
        if right_hand.y > 0.99:
            right_hand.y = 0.99
        
        # Center coordinates 
        middle_shoulder = (int(center_shoulder_x * 1280 * 0.72) + 160, int((center_shoulder_y - 0.01) * 720 * 0.72) + 100)
        left_center_coordinates = (int(left_hand.x * 1280 * 0.72) + 160, int((left_hand.y - 0.01) * 720 * 0.72) + 100)
        right_center_coordinates = (int(right_hand.x * 1280 * 0.72) + 160, int((right_hand.y - 0.01) * 720 * 0.72) + 100)

        # Fix boundary error
        body_detect_y_range = 400
        if middle_shoulder[1] + body_detect_y_range >= 720:
            body_detect_y_range = 720 - middle_shoulder[1]

        # Flatten the 6x6 subarray before calculating the median
        body_subarray = [depth_image[i][middle_shoulder[0]-50 : middle_shoulder[0]+50] 
                    for i in range(middle_shoulder[1]-5, middle_shoulder[1]+body_detect_y_range)]
        left_subarray = [depth_image[i][left_center_coordinates[0]-5 : left_center_coordinates[0]+5] 
                    for i in range(left_center_coordinates[1]-5, left_center_coordinates[1]+5)]
        right_subarray = [depth_image[i][right_center_coordinates[0]-5 : right_center_coordinates[0]+5] 
                    for i in range(right_center_coordinates[1]-5, right_center_coordinates[1]+5)]
        
        # Flatten the subarray
        body_depth_array = [item for row in body_subarray for item in row]
        left_depth_array = [item for row in left_subarray for item in row]
        right_depth_array = [item for row in right_subarray for item in row]

        # Calculate the median
        if body_depth_array:
            mean_body_depth.append(statistics.median(body_depth_array))
        if left_depth_array:
            mean_left_depth.append(statistics.median(left_depth_array))
        if right_depth_array: 
            mean_right_depth.append(statistics.median(right_depth_array))


        if len(mean_body_depth) > 25 and len(mean_left_depth) > 25 and len(mean_right_depth) > 25:
            mean_body_depth.pop(0)
            mean_left_depth.pop(0)
            mean_right_depth.pop(0)

        body_depth = statistics.mean(mean_body_depth)
        left_depth = statistics.mean(mean_left_depth)
        right_depth = statistics.mean(mean_right_depth)

        print("delay_time = ",time.time() - pre_time)

        print("body_depth = ", body_depth)
        print("left_depth = ", left_depth)
        print("right_depth = ", right_depth)

        if (body_depth - pre_body_depth) > 250 and run_time == 100:
            body_depth = pre_body_depth
        if (left_depth - pre_left_depth) > 250 and run_time == 100:
            left_depth = pre_left_depth
        if (right_depth - pre_right_depth) > 250 and run_time == 100:
            right_depth = pre_right_depth
    
        print("")
        print("pre_left_depth = ", pre_left_depth)
        print("pre_right_depth = ", pre_right_depth)

        pre_body_depth = body_depth
        pre_left_depth = left_depth
        pre_right_depth = right_depth

        run_time += 1

        # Radius of circle 
        radius = 20
        
        # Blue color in BGR 
        color = (0, 255, 0) 
        
        # Line thickness of 2 px 
        thickness = 20
        
        # Using cv2.circle() method 
        # Draw a circle with blue line borders of thickness of 2 px 
        depth_colormap = cv2.circle(depth_colormap, left_center_coordinates, radius, color, thickness)
        depth_colormap = cv2.circle(depth_colormap, right_center_coordinates, radius, color, thickness)

        if send_delay_counter == 1:

            l_hand_long = ((body_depth - left_depth) / 1000) * long_factor
            r_hand_long = ((body_depth - right_depth) / 1000) * long_factor
            left_hand_high = ((left_center_coordinates[1] - middle_shoulder[1] ) * -0.002) * high_factor + 0.49
            right_hand_high = ((right_center_coordinates[1] - middle_shoulder[1] ) * -0.002) * high_factor + 0.49
            left_hand_width = (left_center_coordinates[0] - middle_shoulder[0] ) * 0.0014 * width_factor
            right_hand_width = (right_center_coordinates[0] - middle_shoulder[0] ) * -0.0014 * width_factor
            body_high = (1 - middle_shoulder[1]) * 0.002 + 1.4

            if not (l_hand_long > -0.0975 and l_hand_long < 0.0975 and left_hand_width < 0.087 and left_hand_width > -0.087 and left_hand_high < 0.49 and left_hand_high > -0.1):
                if not (r_hand_long > -0.0975 and r_hand_long < 0.0975 and right_hand_width < 0.087 and right_hand_width > -0.087 and right_hand_high < 0.49 and right_hand_high > -0.1):
                    movement = [left_hand_high, left_hand_width, l_hand_long, right_hand_high, right_hand_width, r_hand_long, body_high]
                    movement_list = [f"{point:.2f}" for point in movement]

            send_delay_counter = 0

        print("send_delay_counter = ",send_delay_counter)

        send_delay_counter += 1

        print("delay_time = ",time.time() - pre_time)
        
        print("movement_list = ",movement_list)
        
        send_data(movement_list, host, port)

        # Display the output
        cv2.imshow('MediaPipe Pose', color_image)

        #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        #cv2.imshow('RealSense', depth_colormap)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the webcam and close windows
pipeline.stop()
cv2.destroyAllWindows()
