# This code example demonstrates how to use the ZED 2i stereo camera to capture images and depth data,
# displaying them using OpenCV, with comments in Turkish.
import pyzed.sl as sl
import cv2
import numpy as np

zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.coordinate_units = sl.UNIT.METER
init_params.depth_mode = sl.DEPTH_MODE.NEURAL 

status = zed.open(init_params)

if status != sl.ERROR_CODE.SUCCESS:
    print(f"Kamera açılamadı: {status}")
    exit(1)
print("Kamera başarıyla açıldı!")

runtime_params = sl.RuntimeParameters()
image = sl.Mat()
depth = sl.Mat()

while True:
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        img = image.get_data() # ZED Mat objesini NumPy dizisine çevirir, OpenCV ile işleyebilmek için.

        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        depth_data = depth.get_data()

        depth_data = np.nan_to_num(depth_data, nan=0.0, posinf=0.0, neginf=0.0)

        depth_normalized = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = depth_normalized.astype(np.uint8)

        cv2.imshow("ZED Sol Kamera", img)
        cv2.imshow("ZED Derinlik", depth_normalized)

        print("Depth min:", depth_data.min(), "max:", depth_data.max())

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()
zed.close()
