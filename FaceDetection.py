# Based on code form https://github.com/JetsonHacksNano/CSI-Camera
# MIT License
# Copyright (c) 2019 JetsonHacks
# Copyright (c) 2020 Stefan Larsson
# See LICENSE for OpenCV license and additional information


import cv2
from modules.csi_camera import CSI_Camera
from sys import platform
from size2dist import Size2Dist

show_fps = True
MIN_FACE_SIZE = 1.1
MAX_FACE_SIZE = 6
ON_LINUX = platform == 'linux'

# create size to dist conversion function
size2dist = Size2Dist(n_polynomials=5, interpolate=True)


# Simple draw label on an image; in our case, the video frame
def draw_label(cv_image, label_text, label_position, scale=0.5, color=(255, 255, 255),
               font_face=cv2.FONT_HERSHEY_SIMPLEX
               ):
    # You can get the size of the string with cv2.getTextSize here
    cv2.putText(cv_image, label_text, label_position, font_face, scale, color, 1, cv2.LINE_AA)


# Read a frame from the camera, and draw the FPS on the image if desired
# Return an image
def read_camera(csi_camera, display_fps):
    _, camera_image = csi_camera.read()
    if display_fps:
        draw_label(camera_image, "Frames Displayed (PS): " + str(csi_camera.last_frames_displayed), (10, 20))
        draw_label(camera_image, "Frames Read (PS): " + str(csi_camera.last_frames_read), (10, 40))
    return camera_image


# Good for 1280x720
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 360
# For 1920x1080
# DISPLAY_WIDTH=960
# DISPLAY_HEIGHT=540

# 1920x1080, 30 fps
SENSOR_MODE_1080 = 2
# 1280x720, 60 fps
SENSOR_MODE_720 = 3


def face_detect():
    if ON_LINUX:
        face_cascade = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")
        eye_cascade = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_eye.xml")
    else:
        face_cascade = cv2.CascadeClassifier("modules/haarcascade_frontalface_default.xml")
        eye_cascade = cv2.CascadeClassifier("modules/haarcascade_eye.xml")

    left_camera = CSI_Camera()
    left_camera.create_gstreamer_pipeline(
        sensor_id=0,
        sensor_mode=3,
        framerate=30,
        flip_method=0,
        display_height=DISPLAY_HEIGHT,
        display_width=DISPLAY_WIDTH,
    )
    left_camera.open(left_camera.gstreamer_pipeline)
    left_camera.start()

    cv2.namedWindow("Face Detect", cv2.WINDOW_AUTOSIZE)

    if (
            not left_camera.video_capture.isOpened()
    ):
        # Cameras did not open, or no camera attached

        print("Unable to open any cameras")
        # TODO: Proper Cleanup
        SystemExit(0)
    try:
        # Start counting the number of frames read and displayed
        left_camera.start_counting_fps()
        while cv2.getWindowProperty("Face Detect", 0) >= 0:
            img = read_camera(left_camera, False)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, MIN_FACE_SIZE, MAX_FACE_SIZE)
            dist = 0
            for (x, y, w, h) in faces:
                dist = round(float(size2dist(w)), 1)

                print('Face distance: {}'.format(dist))
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y: y + h, x: x + w]
                roi_color = img[y: y + h, x: x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(
                        roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2
                    )
            if show_fps:
                draw_label(img, "Frames Displayed (PS): " + str(left_camera.last_frames_displayed), (10, 20))
                draw_label(img, "Frames Read (PS): " + str(left_camera.last_frames_read), (10, 40))
                draw_label(img, "Dist: " + str(dist) + "m", (10, 120), 2, color=(0, 0, 0))
            cv2.imshow("Face Detect", img)
            left_camera.frames_displayed += 1
            keyCode = cv2.waitKey(5) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break
    finally:
        left_camera.stop()
        left_camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    face_detect()
