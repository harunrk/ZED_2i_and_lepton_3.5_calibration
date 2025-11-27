# This script captures images from a ZED 2i camera and a thermal camera simultaneously.
# Its used to take chessboard photos for calibration purposes.

import pyzed.sl as sl
import cv2
import os
import threading
import time

# Global değişkenler
zed_frame = None
thermal_frame = None
stop_threads = False

OUTPUT_DIR = "photos" 
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def zed_camera_thread():
    global zed_frame, stop_threads

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Hata: ZED kamera açılamadı.")
        return
    
    print("ZED kamera açıldı.")

    try:
        while not stop_threads:
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed_image = sl.Mat()
                zed.retrieve_image(zed_image, sl.VIEW.LEFT) 
                zed_frame = zed_image.get_data() 
            else:
                time.sleep(0.001)  

    finally:
        zed.close()
        print("ZED kamera kapandı.")

def thermal_camera_thread():
    global thermal_frame, stop_threads

    # Termal kamera indeksi
    thermal_camera_index = 2 
    cap_thermal = cv2.VideoCapture(thermal_camera_index)

    if not cap_thermal.isOpened():
        print(f"Hata: Termal kamera {thermal_camera_index} bağlantısı kurulamadı.")
        return

    print("Termal kamera açıldı.")

    try:
        while not stop_threads:
            ret, frame = cap_thermal.read()
            if ret:
                thermal_frame = frame
            else:
                print("Termal kameradan kare okunamadı. Kamera durduruluyor.")
                break

    finally:
        cap_thermal.release()
        print("Termal kamera kapatılıyor.")


def main():
    global zed_frame, thermal_frame, stop_threads

    print("Kameralar başlatılıyor...")

    zed_thread = threading.Thread(target=zed_camera_thread)
    thermal_thread = threading.Thread(target=thermal_camera_thread)

    zed_thread.start()
    thermal_thread.start()

    print("Kameralar başlatıldı")
    
    while True:
        # Görüntüleri döndürme ve gösterme
        rotated_zed_frame = None
        if zed_frame is not None:
            # ZED görüntüsünü 180 derece döndürme
            rotated_zed_frame = cv2.rotate(zed_frame, cv2.ROTATE_180) 
            cv2.imshow("ZED 2i Live", rotated_zed_frame)
        
        rotated_thermal_frame = None
        if thermal_frame is not None:
            # Termal görüntüyü saat yönünde 90 derece döndürme
            rotated_thermal_frame = cv2.rotate(thermal_frame, cv2.ROTATE_90_CLOCKWISE)
            cv2.imshow("Termal Kamera", rotated_thermal_frame)

        # Klavye girdisi kontrolü
        key = cv2.waitKey(1) & 0xFF
        
        # 'q' tuşuna basıldığında çıkış yap
        if key == ord('q'):
            print("Kullanıcı çıkış istedi.")
            break
        
        # 'k' tuşuna basıldığında fotoğraf çek
        elif key == ord('k'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            print("Fotoğraf çekiliyor...")
            
            # ZED karesini kaydet
            if rotated_zed_frame is not None:
                zed_filename = os.path.join(OUTPUT_DIR, f"zed_{timestamp}.png")
                # `rotated_zed_frame` zaten döndürülmüş ve görüntülenmeye hazır kare
                cv2.imwrite(zed_filename, rotated_zed_frame) 
                print(f"ZED görüntüsü kaydedildi: {zed_filename}")
            else:
                print("ZED karesi alınamadığı için kaydedilemedi.")

            # Termal karesini kaydet
            if rotated_thermal_frame is not None:
                thermal_filename = os.path.join(OUTPUT_DIR, f"thermal_{timestamp}.png")
                # `rotated_thermal_frame` zaten döndürülmüş ve görüntülenmeye hazır kare
                cv2.imwrite(thermal_filename, rotated_thermal_frame)
                print(f"Termal görüntüsü kaydedildi: {thermal_filename}")
            else:
                print("Termal karesi alınamadığı için kaydedilemedi.")

    print("Kaynaklar serbest bırakılıyor...")

    stop_threads = True
    zed_thread.join()
    thermal_thread.join()
    
    cv2.destroyAllWindows()
    print("Başarıyla çıkış yapıldı.")

if __name__ == "__main__":
    main()
