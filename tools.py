import numpy as np


def apply_custom_threshold_filter(image):
    # Bước 1: Tính ngưỡng bằng trung bình mức xám của toàn ảnh
    threshold = np.mean(image)

    # Bước 2: Tạo một ảnh mới để lưu kết quả
    output_image = np.zeros_like(image)

    # Bước 3: Áp dụng bộ lọc cho từng pixel
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            pixel_value = image[i, j]
            if pixel_value < threshold:
                new_value = pixel_value * 0.8
            else:
                new_value = pixel_value * 1.125

            # Giới hạn giá trị pixel trong khoảng 0-255
            output_image[i, j] = min(255, max(0, int(new_value)))

    return output_image


def overlay_frame(frame1, frame2, x, y):
    # Kiểm tra số kênh của frame1 và frame2
    if len(frame1.shape) == 2:  # Nếu frame1 là ảnh grayscale (1 kênh)
        frame1 = np.expand_dims(frame1, axis=-1)  # Thêm 1 kênh cuối để biến nó thành 3 kênh
        frame1 = np.repeat(frame1, 3, axis=-1)  # Nhân đôi các kênh để trở thành ảnh RGB

    # Lấy kích thước của frame1
    h1, w1 = frame1.shape[:2]

    # Đảm bảo frame1 nằm hoàn toàn trong khung của frame2
    if y + h1 > frame2.shape[0] or x + w1 > frame2.shape[1]:
        raise ValueError("Frame1 vượt ra ngoài kích thước của Frame2 tại vị trí này.")

    # Vẽ đè frame1 lên frame2 tại tọa độ (x, y)
    frame2[y:y + h1, x:x + w1] = frame1

    return frame2


def bgr_to_gray(frame):
    # Tách các kênh màu BGR
    blue, green, red = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]

    # Áp dụng công thức chuyển đổi sang grayscale
    gray_frame = 0.299 * red + 0.587 * green + 0.114 * blue

    # Chuyển đổi lại thành kiểu dữ liệu uint8 để phù hợp với hình ảnh
    return gray_frame.astype(np.uint8)