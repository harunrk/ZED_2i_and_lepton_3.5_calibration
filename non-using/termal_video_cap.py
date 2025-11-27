import cv2

dosya_adi = 'termal_kayit.avi'
fps = 9

cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Hata: Kamera bağlantısı kurulamadı.")
    exit()

genislik_gercek = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
yukseklik_gercek = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Kamera çözünürlüğü: {genislik_gercek}x{yukseklik_gercek}")

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc(*'MJPG') # Alternatif

out = cv2.VideoWriter(dosya_adi, fourcc, fps, (genislik_gercek, yukseklik_gercek))

print("Video kaydı başlatıldı. Çıkış yapmak için 'q' tuşuna basın.")

while True:
    ret, frame = cap.read()

    if ret:
        cv2.imshow('Termal Kamera', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Kareden veri okunamadı. Kayıt durduruluyor.")
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video kaydı '{dosya_adi}' dosyasına başarıyla kaydedildi.")