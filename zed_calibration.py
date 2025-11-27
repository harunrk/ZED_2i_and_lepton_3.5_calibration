# This code example demonstrates how to calibrate an RGB camera using chessboard images captured from a ZED camera.
# Could be not exactly gives the best results for this camera but it can help to understand how to do camera calibration using OpenCV in Python.
import cv2
import numpy as np
import glob
import os
import sys 

pattern_size = (4, 6)   # Satranç tahtasının iç köşe sayısı 
square_size = 5.5       # mm cinsinden kare kenarı uzunluğu (Kullandığınız tahtanın gerçek ölçüsü)
image_folder = '/home/harunrk/Sensors/Examples/zed_images/' # RGB görüntü klasörü
save_dir = '/home/harunrk/Sensors/Calibration'
image_extension = '*.png' 

objp = np.zeros((np.prod(pattern_size), 3), np.float32)
objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
objp *= square_size  # Gerçek dünya ölçüsüne çevir

obj_points = []  # 3D - Gerçek dünya noktaları listesi
img_points = []  # 2D - Görüntü noktaları listesi

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

images = sorted(glob.glob(os.path.join(image_folder, image_extension)))
print(f"Toplam {len(images)} RGB görüntü bulundu.\n")

cv2.namedWindow('Bulunan Köşeler', cv2.WINDOW_AUTOSIZE)
last_gray_shape = None # Kameranın boyutunu (height, width) tutacak değişken
successful_captures = 0

print("--- KÖŞE TESPİTİ BAŞLADI ---")
for fname in images:
    img = cv2.imread(fname)
    
    if img is None:
        print(f"[HATA] RGB görüntü yüklenemedi: '{fname}'")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        if last_gray_shape is None:
            last_gray_shape = gray.shape # (height, width)
            
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners_refined)
        successful_captures += 1
 
        display = img.copy()
        
        cv2.drawChessboardCorners(display, pattern_size, corners_refined, ret)
        cv2.imshow('Bulunan Köşeler', display)
        
        if cv2.waitKey(100) & 0xFF == 27:
            print("\nKullanıcı isteğiyle köşe bulma durduruldu.")
            break
            
        print(f"[OK] Köşe bulundu: {os.path.basename(fname)} (Toplam: {successful_captures})")
    else:
        print(f"[LOG] Köşe bulunamadı: {os.path.basename(fname)}")

cv2.destroyAllWindows()
print("--- KÖŞE TESPİTİ SONA ERDİ ---\n")

if len(obj_points) > 5: 
    print(f"Kalibrasyon için yeterli görüntü ({len(obj_points)}) toplandı.")
    print("Kamera kalibrasyonu başlatılıyor...")
    
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, last_gray_shape[::-1], None, None
    )

    print("--- RGB KAMERA KALİBRASYONU TAMAMLANDI! ---")
    print("Kamera Matrisi (Intrinsic): ", camera_matrix)
    print("Distorsiyon Katsayıları: ", dist_coeffs)

    total_error = 0
    for i in range(len(obj_points)):
        imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs) 
        error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)  
        total_error += error
        
    mean_error = total_error / len(obj_points)
    print(f"Ortalama Kalibrasyon Hatası: {mean_error:.4f} piksel")
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'rgb_calibration.npz') # Dosya adı değiştirildi

    np.savez(save_path,
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             rvecs=rvecs,
             tvecs=tvecs,
             mean_error=mean_error)

    print(f"\nKalibrasyon dosyası başarıyla kaydedildi: {save_path}")

else:
    print(f"\n[HATA] Kalibrasyon için yeterli sayıda köşe bulunamadı ({len(obj_points)}).")
    print("Lütfen parametreleri ve görüntü kalitesini kontrol edin.")
    print("En az 5-10 başarılı görüntü gereklidir.")
    sys.exit(1)


"""
    Kamera Matrisi (Intrinsic):
    [[891.54441357   0.         585.05285968]
    [  0.         886.649901   373.26475795]
    [  0.           0.           1.        ]]

    Distorsiyon Katsayıları:
    [[ 0.05938586 -1.30656603  0.00417282 -0.01290473  3.94167726]]

    Ortalama Kalibrasyon Hatası: 0.2476 piksel
"""