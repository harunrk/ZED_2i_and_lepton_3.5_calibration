# This script performs stereo calibration between a thermal camera and a ZED camera
# using pre-captured synchronized chessboard images from both cameras.
import cv2
import numpy as np
import glob
import os
import sys

pattern_size = (4, 6)   # Satranç tahtasının iç köşe sayısı
square_size = 5.5       # mm cinsinden kare kenarı uzunluğu

base_dir = '/home/harunrk/Sensors/'
calibration_dir = os.path.join(base_dir, 'Calibration')

image_folder_thermal = os.path.join(base_dir, 'Examples/thermal_images/')
image_folder_zed = os.path.join(base_dir, 'Examples/zed_images/')

ZED_IMAGE_WIDTH = 1280
ZED_IMAGE_HEIGHT = 720
image_size_zed = (ZED_IMAGE_WIDTH, ZED_IMAGE_HEIGHT)

output_file_name = 'stereo_calibration.npz'


# --- 2. İNTRİNSİK PARAMETRELERİN DOSYADAN YÜKLENMESİ ---

def load_calibration_data(file_path):
    """Kaydedilmiş .npz dosyasından kamera matrisi ve distorsiyon katsayılarını yükler."""
    try:
        data = np.load(file_path)
        print(f"[OK] Kalibrasyon verileri yüklendi: {os.path.basename(file_path)}")
        return data['camera_matrix'], data['dist_coeffs']
    except FileNotFoundError:
        print(f"[HATA] Dosya bulunamadı: {file_path}")
        print("Lütfen dosya yolunu veya dosya adını kontrol edin.")
        sys.exit(1)
    except Exception as e:
        print(f"[HATA] Dosya yüklenirken hata oluştu: {e}")
        sys.exit(1)

# Termal Kamera Verilerini Yükle
thermal_calib_path = os.path.join(calibration_dir, 'thermal_calibration.npz')
camera_matrix_thermal, dist_coeffs_thermal = load_calibration_data(thermal_calib_path)

# ZED Kamera Verilerini Yükle
zed_calib_path = os.path.join(calibration_dir, 'rgb_calibration.npz')
camera_matrix_zed, dist_coeffs_zed = load_calibration_data(zed_calib_path)

print("\nYüklü Termal Matrisi:\n", camera_matrix_thermal)
print("\nYüklü ZED Matrisi:\n", camera_matrix_zed)


# --- 3. GÖRÜNTÜ ÇİFTLERİNİN ZAMAN DAMGASINA GÖRE EŞLEŞTİRİLMESİ ---
print("\n--- GÖRÜNTÜ EŞLEŞTİRME BAŞLADI ---")

# Termal görüntüleri zaman damgasına göre sözlüğe kaydet
thermal_files_map = {}
for full_path in glob.glob(os.path.join(image_folder_thermal, 'thermal_*.png')):
    file_name = os.path.basename(full_path)
    # 'thermal_' ve '.png' kısımlarını atarak zaman damgasını (YYYYMMDD_HHmmss) anahtar yap
    key = file_name.replace('thermal_', '').replace('.png', '')
    thermal_files_map[key] = full_path

# ZED görüntülerini gez ve eşleşen termal görüntüleri bul
image_pairs = []
for full_path in glob.glob(os.path.join(image_folder_zed, 'zed_*.png')):
    file_name = os.path.basename(full_path)
    # 'zed_' ve '.png' kısımlarını atarak zaman damgasını al
    key = file_name.replace('zed_', '').replace('.png', '')
    
    # Eşleşme kontrolü
    if key in thermal_files_map:
        image_pairs.append({
            'thermal': thermal_files_map[key],
            'zed': full_path
        })
    # else:
    #     print(f"[LOG] ZED görüntüsü için termal eşleşme bulunamadı: {file_name}")

if not image_pairs:
    print("\n[HATA] Eşleşen senkronize görüntü çifti bulunamadı.")
    sys.exit(1)

print(f"Toplam {len(image_pairs)} senkronize görüntü çifti eşleştirildi.")


# --- 4. KÖŞE TESPİTİ ve NOKTA TOPLAMA ---
objp = np.zeros((np.prod(pattern_size), 3), np.float32)
objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2) * square_size

obj_points = []          # 3D Dünya Noktaları
img_points_thermal = []  # Termal 2D Görüntü Noktaları
img_points_zed = []      # ZED 2D Görüntü Noktaları

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

successful_captures = 0

print("\n--- STEREO KÖŞE TESPİTİ BAŞLADI ---")
for i, pair in enumerate(image_pairs):
    fname_thermal = pair['thermal']
    fname_zed = pair['zed']
    
    # 1. Termal Görüntü İşleme
    img_thermal = cv2.imread(fname_thermal)
    gray_thermal = cv2.cvtColor(img_thermal, cv2.COLOR_BGR2GRAY)
    ret_t, corners_t = cv2.findChessboardCorners(gray_thermal, pattern_size, None)
    
    # 2. ZED Görüntü İşleme
    img_zed = cv2.imread(fname_zed)
    gray_zed = cv2.cvtColor(img_zed, cv2.COLOR_BGR2GRAY)
    ret_z, corners_z = cv2.findChessboardCorners(gray_zed, pattern_size, None)

    # 3. Her İki Kamerada da Köşe Bulunduysa Kaydet
    if ret_t and ret_z:
        corners_t_refined = cv2.cornerSubPix(gray_thermal, corners_t, (5, 5), (-1, -1), criteria)
        corners_z_refined = cv2.cornerSubPix(gray_zed, corners_z, (11, 11), (-1, -1), criteria)
        
        obj_points.append(objp)
        img_points_thermal.append(corners_t_refined)
        img_points_zed.append(corners_z_refined)
        
        successful_captures += 1
        print(f"[OK] Köşe bulundu: Çift {successful_captures}/{len(image_pairs)}")
        
    else:
        print(f"[LOG] Köşe bulunamadı: Çift {os.path.basename(fname_thermal)}")

print(f"\n--- KÖŞE TESPİTİ SONA ERDİ. Toplam başarılı çift: {successful_captures} ---\n")

# --- 5. STEREO KALİBRASYON HESAPLAMA ve KAYDETME ---
if successful_captures > 5:
    print(f"Stereo kalibrasyon için yeterli görüntü ({successful_captures}) toplandı.")
    print("Stereo kalibrasyon başlatılıyor...")

    # Stereo Kalibrasyonu Başlatma
    # cv2.CALIB_FIX_INTRINSIC bayrağı, yüklediğimiz M ve D matrislerini sabitler.
    ret, M1, D1, M2, D2, R, T, E, F = cv2.stereoCalibrate(
        objectPoints=obj_points,
        imagePoints1=img_points_thermal,
        imagePoints2=img_points_zed,
        cameraMatrix1=camera_matrix_thermal,
        distCoeffs1=dist_coeffs_thermal,
        cameraMatrix2=camera_matrix_zed,
        distCoeffs2=dist_coeffs_zed,
        imageSize=image_size_zed,
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    print("\n--- STEREO KALİBRASYON TAMAMLANDI! ---")
    print(f"Reprojection Hatası (Ortalama): {ret:.4f} piksel")
    print("\nDönme Matrisi (R) - Termal'den ZED'e:\n", R)
    print("\nÖteleme Vektörü (T) - Termal'den ZED'e (mm cinsinden):\n", T)

    # Kalibrasyon matrislerini kaydetme (.npz formatında)
    os.makedirs(calibration_dir, exist_ok=True)
    save_path = os.path.join(calibration_dir, output_file_name)

    np.savez(save_path,
             R=R,
             T=T,
             M1=M1, D1=D1,
             M2=M2, D2=D2,
             reprojection_error=ret)

    print(f"\nStereo Kalibrasyon dosyası başarıyla kaydedildi: {save_path}")
    
else:
    print(f"\n[HATA] Stereo kalibrasyon için yeterli sayıda senkronize köşe bulunamadı ({successful_captures}).")
    print("Lütfen görüntüleri ve köşe tespitini kontrol edin. En az 5-10 başarılı görüntü çifti gereklidir.")
    sys.exit(1)