# This code aligns, rectifies, and overlays images from a ZED RGB camera 
# and a thermal camera using calibration data.
# The rectification process aligns the optical axes of both cameras,
# providing a more accurate overlap.
# NOTE: The accuracy of the calibration data and the physical placement 
# of the cameras can affect the quality of the results.

import pyzed.sl as sl
import cv2
import numpy as np
import sys

RGB_CALIB_PATH = "/home/harunrk/Sensors/Calibration/rgb_calibration.npz"
THERMAL_CALIB_PATH = "/home/harunrk/Sensors/Calibration/thermal_calibration.npz"
STEREO_CALIB_PATH = "/home/harunrk/Sensors/Calibration/stereo_calibration.npz"
THERMAL_CAM_ID = 2
WIDTH, HEIGHT = 1280, 720 
SIZE = (WIDTH, HEIGHT)
ALPHA = 0.5

def load_calibration_data():
    """Kalibrasyon verilerini yükler ve matrisleri döndürür."""
    try:
        rgb_calib = np.load(RGB_CALIB_PATH)
        thermal_calib = np.load(THERMAL_CALIB_PATH)
        stereo_calib = np.load(STEREO_CALIB_PATH)
    except FileNotFoundError as e:
        print(f"HATA: Kalibrasyon dosyası bulunamadı: {e.filename}")
        sys.exit(1)

    K_rgb = rgb_calib['camera_matrix']
    D_rgb = rgb_calib['dist_coeffs']
    K_thermal = thermal_calib['camera_matrix']
    D_thermal = thermal_calib['dist_coeffs']
    R = stereo_calib['R']
    T = stereo_calib['T']

    return K_rgb, D_rgb, K_thermal, D_thermal, R, T

def initialize_zed_camera():
    """ZED kamerayı başlatır."""
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"HATA: ZED kamerayı açamadı: {status}")
        sys.exit(1)
    
    return zed

def initialize_thermal_camera():
    """Termal kamerayı başlatır ve çözünürlüğünü ayarlamaya çalışır."""
    cap_thermal = cv2.VideoCapture(THERMAL_CAM_ID)
    if not cap_thermal.isOpened():
        print(f"HATA: Termal kamera (ID: {THERMAL_CAM_ID}) açılamadı.")
        sys.exit(1)
        
    cap_thermal.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap_thermal.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    
    return cap_thermal

def main():
    K_rgb, D_rgb, K_thermal, D_thermal, R, T = load_calibration_data()

    zed = initialize_zed_camera()
    cap_thermal = initialize_thermal_camera()
    runtime_params = sl.RuntimeParameters()

    print("Stereo rektifiye haritaları hesaplanıyor...")
    
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_rgb, D_rgb, K_thermal, D_thermal, SIZE, R, T, flags=cv2.CALIB_ZERO_DISPARITY
    )

    map1_rgb, map2_rgb = cv2.initUndistortRectifyMap(K_rgb, D_rgb, R1, P1, SIZE, cv2.CV_32FC1)
    map1_thermal, map2_thermal = cv2.initUndistortRectifyMap(K_thermal, D_thermal, R2, P2, SIZE, cv2.CV_32FC1)
    
    print("Haritalar hazır. Döngü başlıyor...")
    
    try:
        while True:
            zed_image = sl.Mat()
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(zed_image, sl.VIEW.LEFT) 
                img_rgb = zed_image.get_data()
                
                if img_rgb.shape[-1] == 4:
                    img_rgb = img_rgb[:, :, :3]
                
                img_rgb = cv2.rotate(img_rgb, cv2.ROTATE_180) # ZED Kamerası 180 derece ters takılı

                ret_th, img_th = cap_thermal.read()
                if not ret_th:
                    print("Termal görüntü okunamadı, atlanıyor.")
                    continue
                
                if len(img_th.shape) == 2:  # C. Termal Görüntüyü 3 Kanallı Renk Formatına Çevir
                    img_th_color = cv2.cvtColor(img_th, cv2.COLOR_GRAY2BGR)
                elif len(img_th.shape) == 3 and img_th.shape[2] != 3:
                     img_th_color = img_th[:, :, :3]
                else: 
                    img_th_color = img_th
                
                img_th_color = cv2.rotate(img_th_color, cv2.ROTATE_90_CLOCKWISE) # Termal Kamera 90 derece sağa yatık
                
                # D. Rektifiye İşlemi (Hizalama ve Distorsiyon Giderme)
                # Döndürülmüş ve doğru yöne çevrilmiş ham görüntülerle rektifikasyon uygula.
                img_rgb_rectified = cv2.remap(img_rgb, map1_rgb, map2_rgb, cv2.INTER_LINEAR)
                img_th_rectified = cv2.remap(img_th_color, map1_thermal, map2_thermal, cv2.INTER_LINEAR)
                
                if img_th_rectified.shape != img_rgb_rectified.shape:
                    print(f"UYARI: Rektifiye sonrası boyutlar uyuşmuyor: RGB {img_rgb_rectified.shape} vs TH {img_th_rectified.shape}. Yeniden boyutlandırılıyor.")
                    
                    target_size = (img_rgb_rectified.shape[1], img_rgb_rectified.shape[0])
                    img_th_rectified = cv2.resize(img_th_rectified, target_size, interpolation=cv2.INTER_LINEAR)
                
                overlay = cv2.addWeighted(img_rgb_rectified, 1 - ALPHA, img_th_rectified, ALPHA, 0)

                cv2.imshow("RGB + Thermal Overlay (Rectified)", overlay)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("\nProgram sonlandırılıyor...")
        cap_thermal.release()
        zed.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()