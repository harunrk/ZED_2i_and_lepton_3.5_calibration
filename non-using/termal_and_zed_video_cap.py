import pyzed.sl as sl
import cv2
import os
import threading
import time

# Ortak değişkenler
zed_frame = None
thermal_frame = None
zed_recording_enabled = False
thermal_recording_enabled = False
stop_threads = False

# ZED 2i Kamera ve Kayıt İş Parçacığı
def zed_camera_thread():
    global zed_frame, zed_recording_enabled, stop_threads

    zed_folder = "ZED_2i_video"
    os.makedirs(zed_folder, exist_ok=True)
    zed_video_path = os.path.join(zed_folder, "video_cap.svo")

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Hata: ZED kamera açılamadı.")
        return

    zed_rec_params = sl.RecordingParameters(zed_video_path, sl.SVO_COMPRESSION_MODE.H264)
    if zed.enable_recording(zed_rec_params) != sl.ERROR_CODE.SUCCESS:
        print("Hata: ZED video kaydı başlatılamadı.")
        return

    zed_recording_enabled = True
    print("ZED kamera kaydı başlatıldı.")

    try:
        while not stop_threads:
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed_image = sl.Mat()
                zed.retrieve_image(zed_image, sl.VIEW.LEFT)
                zed_frame = zed_image.get_data()
            else:
                time.sleep(0.001) 

    finally:
        if zed_recording_enabled:
            zed.disable_recording()
        zed.close()
        print("ZED kamera kaydı tamamlandı.")

# Termal Kamera ve Kayıt İş Parçacığı
def thermal_camera_thread():
    global thermal_frame, thermal_recording_enabled, stop_threads

    thermal_folder = "Thermal_video"
    os.makedirs(thermal_folder, exist_ok=True)
    thermal_video_path = os.path.join(thermal_folder, "termal_kayit.avi")

    thermal_camera_index = 2
    cap_thermal = cv2.VideoCapture(thermal_camera_index)

    if not cap_thermal.isOpened():
        print(f"Hata: Termal kamera {thermal_camera_index} bağlantısı kurulamadı.")
        return

    fps = 9
    genislik_gercek = int(cap_thermal.get(cv2.CAP_PROP_FRAME_WIDTH))
    yukseklik_gercek = int(cap_thermal.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out_thermal = cv2.VideoWriter(thermal_video_path, fourcc, fps, (genislik_gercek, yukseklik_gercek))

    thermal_recording_enabled = True
    print("Termal kamera kaydı başlatıldı.")

    try:
        while not stop_threads:
            ret, frame = cap_thermal.read()
            if ret:
                thermal_frame = frame
                out_thermal.write(frame)
            else:
                print("Termal kameradan kare okunamadı. Kayıt durduruluyor.")
                break

    finally:
        cap_thermal.release()
        out_thermal.release()
        print("Termal kamera kaydı tamamlandı.")

def main():
    global zed_frame, thermal_frame, stop_threads

    print("Kamera kayıtları başlatılıyor...")
    
    zed_thread = threading.Thread(target=zed_camera_thread)
    thermal_thread = threading.Thread(target=thermal_camera_thread)

    zed_thread.start()
    thermal_thread.start()

    print("Kayıtlar başladı. Çıkmak için 'q' tuşuna basın.")
    
    while True:
        if zed_frame is not None:
            cv2.imshow("ZED 2i Live", zed_frame)
        
        if thermal_frame is not None:
            cv2.imshow("Termal Kamera", thermal_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Kullanıcı çıkış istedi.")
            break
        
    print("Kayıtlar tamamlanıyor, kaynaklar serbest bırakılıyor...")

    stop_threads = True
    zed_thread.join()
    thermal_thread.join()
    
    cv2.destroyAllWindows()
    print("Video kayıtları başarıyla tamamlandı.")

if __name__ == "__main__":
    main()