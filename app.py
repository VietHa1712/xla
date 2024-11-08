import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tools import apply_custom_threshold_filter
from tools import overlay_frame

# 1. Load các model đã huấn luyện
emotion_model = load_model('emotion_detection_model.h5')
age_gender_model = load_model('age_gender_model.keras')

# Các nhãn cảm xúc tương ứng
emotion_labels = {
    0: 'Anger',
    1: 'Disgust',
    2: 'Neutral',
    3: 'Happiness',
    4: 'Sadness',
    5: 'Surprise',
    6: 'Neutral',
    7: 'Neutral'
}

# Load Haar Cascade để phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Mở camera
cap = cv2.VideoCapture(0)

# Biến đếm khung hình (frames)
frame_count = 0
prediction_interval = 10
info_text = ''

while True:
    # 3. Đọc khung hình từ camera
    ret, frame = cap.read()
    if not ret:
        break

    # Phản chiếu khung hình trước khi xử lý
    frame = cv2.flip(frame, 1)  # Phản chiếu khung hình theo chiều ngang

    # Chuyển ảnh sang màu xám
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    filtered_frame = gray_frame.copy()

    # Phát hiện khuôn mặt trong ảnh đã được lọc
    faces = face_cascade.detectMultiScale(filtered_frame, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120))

    # Vẽ hộp bao quanh khuôn mặt và dự đoán cảm xúc, tuổi, giới tính trên ảnh đã lọc
    for (x, y, w, h) in faces:
        face = filtered_frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (48, 48))
        face_resized = apply_custom_threshold_filter(face_resized)

        face_display = cv2.resize(face_resized, (100, 100))
        frame = overlay_frame(face_display, frame, 0, 0)

        # Tăng biến đếm khung hình
        frame_count += 1

        # Dự đoán chỉ khi frame_count đạt điều kiện
        if frame_count % prediction_interval == 0:
            # Chuẩn hóa và reshape ảnh để phù hợp với đầu vào của model cảm xúc
            face_normalized = face_resized / 255.0
            face_normalized = np.reshape(face_normalized, (1, 48, 48, 1))

            # 4. Dự đoán cảm xúc
            emotion_prediction = emotion_model.predict(face_normalized, verbose=0)
            emotion_index = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_index]

            # Chuẩn bị ảnh cho model tuổi và giới tính (chuẩn hóa và reshape về 48x48x1)
            age_gender_input = np.reshape(face_normalized, (1, 48, 48, 1))

            # 5. Dự đoán tuổi và giới tính
            age_pred, gender_pred = age_gender_model.predict(age_gender_input, verbose=0)
            age_text = str(int(age_pred[0][0] * 0.85))  # Chuyển dự đoán tuổi về dạng chuỗi
            gender_text = "Male" if gender_pred[0][0] < 0.5 else "Female"

            # 6. Hiển thị cảm xúc, tuổi và giới tính lên màn hình
            info_text = f"Age: {age_text}\nGender: {gender_text}\n{emotion_text}"

        y0, dy = y - 50, 20  # Điều chỉnh vị trí chữ ban đầu và khoảng cách dòng
        for i, line in enumerate(info_text.split('\n')):
            y_text = y0 + i * dy
            # Vẽ nền đen cho text
            (text_w, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y_text - text_h), (x + text_w, y_text), (0, 0, 0), -1)
            # Hiển thị chữ trắng trên nền đen với kích thước lớn hơn
            cv2.putText(frame, line, (x, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # Vẽ hộp bao quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Hiển thị khung hình
    cv2.imshow('Face analysis', frame)

    # Nhấn 'ESC' để thoát
    if cv2.waitKey(1) & 0xFF == 27:  # 27 là mã ASCII cho phím 'ESC'
        break

# Giải phóng camera và đóng tất cả cửa sổ
cap.release()
cv2.destroyAllWindows()
