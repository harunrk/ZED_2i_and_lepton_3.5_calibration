# This code example demonstrates how to calibrate an RGB camera using chessboard images captured from a ZED camera.
# Could be not exactly gives the best results for this camera but it can help to understand how to do camera calibration using OpenCV in Python.
import cv2
import numpy as np
import glob
import os
import sys 

pattern_size = (4, 6)   # Satranç tahtasının iç köşe sayısı (Örn: 5x7 kare için (4, 6) yazılır)
square_size = 5.5       # mm cinsinden kare kenarı uzunluğu (Kullandığınız tahtanın gerçek ölçüsü)

image_folder = '/home/harunrk/Sensors/Examples/thermal_images/'  
save_dir = '/home/harunrk/Sensors/Calibration'

objp = np.zeros((np.prod(pattern_size), 3), np.float32)
objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
objp *= square_size  # Gerçek dünya ölçüsüne çevir

obj_points = []  # 3D - Gerçek dünya noktaları listesi
img_points = []  # 2D - Görüntü noktaları listesi

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

images = sorted(glob.glob(os.path.join(image_folder, '*.png')))
print(f"Toplam {len(images)} termal görüntü bulundu.\n")

cv2.namedWindow('Bulunan Köşeler', cv2.WINDOW_AUTOSIZE)
last_gray_shape = None # Kameranın boyutunu (height, width) tutacak değişken
successful_captures = 0

print("--- KÖŞE TESPİTİ BAŞLADI ---")
for fname in images:
    img = cv2.imread(fname)
    
    if img is None:
        print(f"[HATA] Termal görüntü yüklenemedi: '{fname}'")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        if last_gray_shape is None:
            last_gray_shape = gray.shape # (height, width)
            
        corners_refined = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        
        obj_points.append(objp)
        img_points.append(corners_refined)
        successful_captures += 1
 
        display = img.copy()
 
        cv2.drawChessboardCorners(display, pattern_size, corners_refined, ret)
        cv2.imshow('Bulunan Köşeler', display)
        # ESC (27) tuşuna basılırsa döngüden çık
        if cv2.waitKey(100) & 0xFF == 27:
            print("\nKullanıcı isteğiyle köşe bulma durduruldu.")
            break
            
        print(f"[OK] Köşe bulundu: {os.path.basename(fname)} (Toplam: {successful_captures})")
    else:
        print(f"[LOG] Köşe bulunamadı: {os.path.basename(fname)}")

cv2.destroyAllWindows()
print("--- KÖŞE TESPİTİ SONA ERDİ ---\n")

# --- KALİBRASYON HESAPLAMA ve KAYDETME ---
if len(obj_points) > 5: 
    print(f"Kalibrasyon için yeterli görüntü ({len(obj_points)}) toplandı.")
    print("Kamera kalibrasyonu başlatılıyor...")
    
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, last_gray_shape[::-1], None, None
    )

    print("--- TERMAL KAMERA KALİBRASYONU TAMAMLANDI! ---")
    print("\nKamera Matrisi (Intrinsic):\n", camera_matrix)
    print("\nDistorsiyon Katsayıları:\n", dist_coeffs)

    total_error = 0
    for i in range(len(obj_points)):
        # 3D noktaları, bulunan matrislerle 2D görüntüye yansıt
        imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs) 
        # Yansıtılan ve tespit edilen noktalar arasındaki ortalama farkı bul
        error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)  
        total_error += error
        
    mean_error = total_error / len(obj_points)
    print(f"\nOrtalama Kalibrasyon Hatası: {mean_error:.4f} piksel")
    
    # Kalibrasyon matrislerini kaydetme (.npz formatında)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'thermal_calibration.npz')

    np.savez(save_path,
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             rvecs=rvecs,
             tvecs=tvecs,
             mean_error=mean_error)

    print(f"\nKalibrasyon dosyası başarıyla kaydedildi: {save_path}")

else:
    print(f"\n[HATA] Kalibrasyon için yeterli sayıda köşe bulunamadı ({len(obj_points)}).")
    print("Lütfen pattern_size, görüntü kalitesini veya image_folder yolunu kontrol edin.")
    print("En az 5-10 başarılı görüntü gereklidir.")
    sys.exit(1)



"""
    Kamera Matrisi (Intrinsic):
    [[134.05979245   0.          42.94403653]
    [  0.         134.84558416  91.21760684]
    [  0.           0.           1.        ]]

    Distorsiyon Katsayıları:
    [[-0.14476131 -0.20249357 -0.02874295  0.00546302  1.07095395]]

    Ortalama Kalibrasyon Hatası: 0.1475 piksel
"""
















# import cv2
# import numpy as np

# single_image_name = '/home/harunrk/Sensors/Examples/thermal_images/thermal_20251006_103632.png'
# pattern_size = (4, 6) # Termal deseninizin iç köşe sayısı (Örn: 5x3 kare için (4, 2) veya 6x4 kare için (5, 3))
# window_name = 'Termal - Bulunan Koseler'
# img = cv2.imread(single_image_name)

# if img is None:
#     print(f"HATA: '{single_image_name}' dosyası bulunamadı veya yüklenemedi. Lütfen dosya yolunu kontrol edin.")
# else:
#     if len(img.shape) == 3: # 1. Görüntüyü Gri Tonlamaya Çevir
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         img_display = img.copy() # Orijinal renkli kopyayı çizim için tut

#     ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

#     if ret:
#         print(f"BAŞARILI: {pattern_size} boyutunda köşeler bulundu!")

#         criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#         corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        
#         radius = 1 # 160x120 çözünürlük için önerilen köşe çizim yarıçapı
#         color = (0, 255, 0) # Yeşil
#         thickness = -1      # İçini doldur

#         for corner in corners.reshape(-1, 2):
#             center_x, center_y = int(corner[0]), int(corner[1])
#             cv2.circle(img_display, (center_x, center_y), radius, color, thickness)

#         cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#         cv2.imshow(window_name, img_display)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#     else:
#         print(f"BAŞARISIZ: Belirtilen boyutta {pattern_size} satranç tahtası köşeleri bulunamadı.")