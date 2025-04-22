import os
# Thêm thư mục stylegan3 vào sys.path để Python có thể tìm thấy các mô-đun của nó
import sys
sys.path.append(os.path.join(os.getcwd(), 'stylegan2-ada-pytorch'))

import torch
import dnnlib
import legacy
import cv2
import numpy as np
import logging

# Thiết lập logging để ghi lại quá trình
logging.basicConfig(filename='process_log.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Đường dẫn tới mô hình pretrained StyleGAN2-ADA-PyTorch
network_pkl = 'models/ffhq.pkl'  # Đảm bảo mô hình pretrained của bạn đúng
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load mô hình StyleGAN2-ADA
def load_model():
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # Generator
    logging.info("Mô hình StyleGAN2-ADA đã được tải thành công.")
    return G

# Hàm để tải ảnh và chuyển đổi thành tensor
def resize_and_convert_to_tensor(image_path, target_size=(128, 128)):
    """
    Đọc ảnh, resize về kích thước target_size và chuyển đổi thành tensor.
    """
    image = cv2.imread(image_path)
    
    # Kiểm tra nếu ảnh có 4 kênh (RGBA), nếu có thì bỏ alpha channel
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    h, w, _ = image.shape

    # Tính toán tỷ lệ resize sao cho một chiều có kích thước target_size, giữ nguyên tỷ lệ
    if h < w:
        new_w = target_size[0]
        new_h = int(h * (target_size[0] / w))
    else:
        new_h = target_size[1]
        new_w = int(w * (target_size[1] / h))

    # Resize ảnh theo tỷ lệ giữ nguyên kích thước
    image_resized = cv2.resize(image, (new_w, new_h))

    # Padding (thêm pixel trắng) nếu cần để đạt kích thước target_size
    top = (target_size[1] - new_h) // 2
    bottom = target_size[1] - new_h - top
    left = (target_size[0] - new_w) // 2
    right = target_size[0] - new_w - left

    image_padded = cv2.copyMakeBorder(image_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # Kiểm tra kích thước ảnh trước khi chuyển đổi
    logging.info(f"Ảnh sau khi resize và padding: {image_padded.shape}")

    # Chuyển ảnh thành tensor và chuẩn hóa giá trị từ [0, 255] -> [0, 1]
    image_tensor = torch.Tensor(image_padded).permute(2, 0, 1).unsqueeze(0) / 255.0  # Từ (H, W, C) sang (C, H, W)

    logging.info(f"Tensor của ảnh: {image_tensor.shape}")
    return image_tensor

def generate_child_image(father_image_path, mother_image_path, age_factor, gender, G):
    """
    Sinh ảnh con từ ảnh của bố và mẹ với độ tuổi và giới tính được chỉ định.
    """
    try:
        logging.info(f"Đang xử lý ảnh của bố: {father_image_path} và mẹ: {mother_image_path}...")

        # Đọc và chuẩn hóa ảnh của bố và mẹ (giảm kích thước xuống 128x128)
        father_image = resize_and_convert_to_tensor(father_image_path, target_size=(128, 128))
        mother_image = resize_and_convert_to_tensor(mother_image_path, target_size=(128, 128))

        # Kiểm tra kích thước của các tensor
        logging.info(f"Ảnh của bố: {father_image.shape}, Ảnh của mẹ: {mother_image.shape}")

        # Tạo latent vector cho cả bố và mẹ
        latent_father = torch.randn(1, 512).to(device)  # Latent vector ngẫu nhiên cho bố
        latent_mother = torch.randn(1, 512).to(device)  # Latent vector ngẫu nhiên cho mẹ

        # Tạo điều kiện 'c' cho độ tuổi và giới tính
        c = torch.randn(2, 512).to(device)  # Điều kiện ngẫu nhiên cho bố và mẹ
        if gender == "Female":
            c = c * 1.2  # Tăng cường độ sáng/đặc trưng cho nữ (có thể thay đổi tùy theo yêu cầu)

        # Gộp latent vectors của bố và mẹ thành một tensor duy nhất
        latent_input = torch.cat([latent_father, latent_mother], dim=0)  # Gộp theo chiều batch

        # Sinh ảnh con từ các latent vector và điều chỉnh độ tuổi
        with torch.no_grad():
            generated_image = G(latent_input, c)  # Cung cấp latent vectors cho bố và mẹ và điều kiện 'c'
            generated_image = generated_image.squeeze().cpu().numpy().transpose(1, 2, 0)

            # Điều chỉnh độ tuổi của con theo age_factor
            age_adjusted_image = adjust_age(generated_image, age_factor)

        # Lưu ảnh con đã sinh ra vào thư mục output
        output_image_path = os.path.join('images/output', f'generated_child_{age_factor}_{gender}.jpg')
        cv2.imwrite(output_image_path, cv2.cvtColor(age_adjusted_image, cv2.COLOR_RGB2BGR))

        logging.info(f"Ảnh con ở độ tuổi {age_factor} và giới tính {gender} đã được sinh ra và lưu vào {output_image_path}.")
        return output_image_path
    except Exception as e:
        logging.error(f"Xảy ra lỗi trong quá trình xử lý ảnh: {e}")
        raise e

# Hàm điều chỉnh độ tuổi của con
def adjust_age(image, age_factor):
    """
    Điều chỉnh ảnh con theo độ tuổi (mẫu, có thể được tùy chỉnh thêm).
    """
    try:
        logging.info(f"Điều chỉnh ảnh theo độ tuổi {age_factor}.")
        # Giả lập việc điều chỉnh độ tuổi (có thể thay đổi thêm tùy theo yêu cầu)
        age_adjusted_image = image  # Đây là placeholder, có thể thêm logic điều chỉnh độ tuổi tại đây
        return age_adjusted_image
    except Exception as e:
        logging.error(f"Xảy ra lỗi khi điều chỉnh độ tuổi: {e}")
        raise e
