# This code aligns, rectifies, and overlays images from a ZED RGB camera 
# and a thermal camera using calibration data.
# The rectification process aligns the optical axes of both cameras,
# providing a more accurate overlap.
# NOTE: The accuracy of the calibration data and the physical placement 
# of the cameras can affect the quality of the results.

import cv2
import numpy as np
import sys
import os 

RGB_CALIB_PATH = "/home/harunrk/Sensors/Calibration/rgb_calibration.npz"
THERMAL_CALIB_PATH = "/home/harunrk/Sensors/Calibration/thermal_calibration.npz"
STEREO_CALIB_PATH = "/home/harunrk/Sensors/Calibration/stereo_calibration.npz"

WIDTH, HEIGHT = 1280, 720 
SIZE = (WIDTH, HEIGHT)
ALPHA = 0.5 # Üst üste bindirme (overlay) saydamlık katsayısı

def load_calibration_data():
    """Kalibrasyon verilerini yükler ve matrisleri döndürür."""
    try:
        rgb_calib = np.load(RGB_CALIB_PATH)
        thermal_calib = np.load(THERMAL_CALIB_PATH)
        stereo_calib = np.load(STEREO_CALIB_PATH)
    except FileNotFoundError as e:
        print(f"HATA: Kalibrasyon dosyası bulunamadı: {e.filename}")
        sys.exit(1)
    except Exception as e:
        print(f"HATA: Kalibrasyon dosyaları yüklenirken bir sorun oluştu: {e}")
        sys.exit(1)

    K_rgb = rgb_calib['camera_matrix']
    D_rgb = rgb_calib['dist_coeffs']
    K_thermal = thermal_calib['camera_matrix']
    D_thermal = thermal_calib['dist_coeffs']
    R = stereo_calib['R'] # Termal'den RGB'ye dönüş rotasyon matrisi
    T = stereo_calib['T'] # Termal'den RGB'ye dönüş öteleme vektörü

    return K_rgb, D_rgb, K_thermal, D_thermal, R, T

def load_images(rgb_path, thermal_path):
    """Görüntüleri yükler ve boyutlandırır."""
    if not os.path.exists(rgb_path) or not os.path.exists(thermal_path):
        print("HATA: Görüntü dosyaları bulunamadı.")
        sys.exit(1)
    
    rgb_img = cv2.imread(rgb_path)
    if rgb_img is None:
        print(f"HATA: RGB görüntü yüklenemedi: {rgb_path}")
        sys.exit(1)

    rgb_img = cv2.resize(rgb_img, SIZE, interpolation=cv2.INTER_LINEAR)
    thermal_img_color = cv2.imread(thermal_path, cv2.IMREAD_COLOR) 
    
    if thermal_img_color is None:
        print(f"HATA: Termal görüntü yüklenemedi: {thermal_path}")
        sys.exit(1)
        
    if len(thermal_img_color.shape) != 3 or thermal_img_color.shape[2] != 3:
        print(f"UYARI: Termal görüntü 3 kanallı (RGB) değil. Yüklenen şekil: {thermal_img_color.shape}")
        if len(thermal_img_color.shape) == 2:
            thermal_img_color = cv2.cvtColor(thermal_img_color, cv2.COLOR_GRAY2BGR)
    
    return rgb_img, thermal_img_color, thermal_img_color.shape[1], thermal_img_color.shape[0]

def project_thermal_to_rgb(thermal_img, K_rgb, D_rgb, K_thermal, D_thermal, R, T, rgb_size):
    h_thermal, w_thermal = thermal_img.shape[:2]
    thermal_points_2d = np.array([[x, y] for y in range(h_thermal) for x in range(w_thermal)], dtype=np.float32)
    
    thermal_undistorted_points = cv2.undistortPoints(
        thermal_points_2d, 
        K_thermal, 
        D_thermal, 
        P=K_thermal # Bu, normalize edilmiş 3D noktalarını tekrar K_thermal ile ölçeklendirir.
    ).squeeze() 

    pts_thermal_cam = np.hstack((thermal_undistorted_points, np.ones((thermal_undistorted_points.shape[0], 1))))
    pts_rgb_cam = np.dot(R, pts_thermal_cam.T).T + T.T 

    r_vec = np.zeros((3, 1))
    t_vec = np.zeros((3, 1))
    
    image_points, _ = cv2.projectPoints(
        pts_rgb_cam,        # RGB kamera koordinat sistemindeki 3D noktalar
        r_vec,              # Rotasyon vektörü (0)
        t_vec,              # Öteleme vektörü (0)
        K_rgb,              # RGB kamera matrisi
        D_rgb               # RGB distorsiyon katsayıları
    )
    
    image_points = image_points.squeeze()

    map_x = image_points[:, 0].reshape((h_thermal, w_thermal)).astype(np.float32)
    map_y = image_points[:, 1].reshape((h_thermal, w_thermal)).astype(np.float32)

    thermal_projected = cv2.remap(
        thermal_img, 
        map_x, 
        map_y, 
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0) # Dışarıdaki alanlar siyah (0) olsun
    )
    
    thermal_projected_resized = cv2.resize(thermal_projected, rgb_size, interpolation=cv2.INTER_LINEAR)

    return thermal_projected_resized

def run_overlay_fusion(rgb_img_path, thermal_img_path):
    """Görüntüleri yükler, yansıtır ve üst üste bindirir."""
    
    print("1. Kalibrasyon verileri yükleniyor...")
    K_rgb, D_rgb, K_thermal, D_thermal, R, T = load_calibration_data()
    
    print("2. Görüntüler yükleniyor (Termal, 3 kanallı (RGB) olarak varsayılıyor)...")
    rgb_img, thermal_img_color, thermal_w, thermal_h = load_images(rgb_img_path, thermal_img_path)
    
    rgb_size = (rgb_img.shape[1], rgb_img.shape[0])

    print("3. Termal görüntü, RGB düzlemine yansıtılıyor (projeksiyon)...")
    thermal_projected = project_thermal_to_rgb(
        thermal_img_color, 
        K_rgb, D_rgb, K_thermal, D_thermal, R, T, 
        rgb_size
    )
    
    if thermal_projected.shape != rgb_img.shape:
        print("HATA: Projeksiyon sonrası termal görüntü boyutu RGB görüntü boyutuyla uyuşmuyor.")
        print(f"RGB Boyutu: {rgb_img.shape}, Projeksiyon Boyutu: {thermal_projected.shape}")
        return

    print("4. Görüntüler üst üste bindiriliyor (overlay)...")

    fused_image = cv2.addWeighted(rgb_img, 1.0 - ALPHA, thermal_projected, ALPHA, 0)

    print("5. Sonuç gösteriliyor...")
    cv2.imshow("RGB ve Termal Füzyon", fused_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("İşlem tamamlandı.")

RGB_IMAGE_PATH = "/home/harunrk/Sensors/Examples/zed_images/zed_20251006_103617.png"     # ZED 2i Sol Görüntü
THERMAL_IMAGE_PATH = "/home/harunrk/Sensors/Examples/thermal_images/thermal_20251006_103617.png" # Renkli Termal Görüntü

if __name__ == "__main__":
    if os.path.exists(RGB_IMAGE_PATH) and os.path.exists(THERMAL_IMAGE_PATH):
        run_overlay_fusion(RGB_IMAGE_PATH, THERMAL_IMAGE_PATH)
    else:
        print("\nLÜTFEN RGB_IMAGE_PATH ve THERMAL_IMAGE_PATH değişkenlerini kendi görüntü dosyası yollarınla güncelle.")


