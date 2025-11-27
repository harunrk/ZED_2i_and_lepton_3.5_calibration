import pyzed.sl as sl
import cv2
import os


video_folder = "ZED_2i_video"   # Kayıt dizini
os.makedirs(video_folder, exist_ok=True)    # Klasör dizinini oluştur
video_path = os.path.join(video_folder, "video_cap_zedddd.svo")  # SVO video kaydı

zed = sl.Camera()

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 15
init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Opsiyonel

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("Kamera açılamadı")
    exit()

# SVO video kaydı başlat
rec_params = sl.RecordingParameters(video_path, sl.SVO_COMPRESSION_MODE.H264)
if zed.enable_recording(rec_params) != sl.ERROR_CODE.SUCCESS:
    print("Video kaydı başlatılamadı. Dizini ve izinleri kontrol et.")
    zed.close()
    exit()

# Görüntü nesnesi
image = sl.Mat()

print("Kamera çalışıyor, SVO video kaydediliyor ve ekranda gösteriliyor... Çıkmak için 'q' tuşuna bas")

try:
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()

            # Ekranda göster
            cv2.imshow("ZED 2i Live", frame)

            # 'q' tuşuna basınca çık
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
except KeyboardInterrupt:
    print("Kayıt durduruluyor...")

# Kaydı bitir, kamerayı kapat ve pencereleri kapat
zed.disable_recording()
zed.close()
cv2.destroyAllWindows()
print("SVO video kaydı tamamlandı.")
