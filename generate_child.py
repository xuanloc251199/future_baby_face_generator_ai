import os
# Thêm thư mục stylegan2-ada-pytorch vào sys.path để Python có thể tìm thấy các mô-đun của nó
import sys
sys.path.append(os.path.join(os.getcwd(), 'stylegan2-ada-pytorch'))
import torch
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from generate import generate_images  # Import hàm generate_images từ StyleGAN2-ADA-PyTorch
import dnnlib
import legacy

# Khởi tạo các mô hình cần thiết
mtcnn = MTCNN(keep_all=True)
inception = InceptionResnetV1(pretrained='vggface2').eval()

# Hàm nhận diện khuôn mặt và trích xuất đặc điểm
def get_face_embedding(image_path):
    img = Image.open(image_path)
    faces = mtcnn(img)  # Trả về danh sách khuôn mặt
    if faces is not None:
        embeddings = inception(faces)
        return embeddings
    return None

# Hàm tạo ảnh con với StyleGAN2-ADA-PyTorch
def generate_child_image(father_image, mother_image, age):
    # Trích xuất đặc điểm khuôn mặt từ ảnh của bố và mẹ
    father_embedding = get_face_embedding(father_image)
    mother_embedding = get_face_embedding(mother_image)

    if father_embedding is None or mother_embedding is None:
        print("Không thể trích xuất đặc điểm khuôn mặt")
        return None

    # Kết hợp đặc điểm khuôn mặt của bố và mẹ (trung bình)
    child_embedding = (father_embedding + mother_embedding) / 2

    # Tạo latent vector từ đặc điểm khuôn mặt đã kết hợp
    latent_vector = torch.randn(1, 512)  # Kích thước latent vector cho StyleGAN2-ADA
    latent_vector[0, :child_embedding.shape[1]] = child_embedding.squeeze(0)  # Gán đặc điểm khuôn mặt vào latent vector

    # Đảm bảo rằng thư mục lưu kết quả ảnh đã tồn tại
    output_dir = "data/generated_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Chỉ định mô hình đã huấn luyện sẵn
    network_path = "models/ffhq.pkl"  # Đường dẫn tới mô hình đã huấn luyện sẵn (FFHQ model)

    # Tải mô hình đã huấn luyện sẵn từ network_path
    print(f"Loading networks from {network_path}...")
    device = torch.device('cpu')  # Dùng CPU thay vì GPU
    with dnnlib.util.open_url(network_path) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # Tải generator (G) từ mô hình đã huấn luyện

    # Tạo ảnh con sử dụng StyleGAN2-ADA
    label = torch.zeros([1, G.c_dim], device=device)  # Không sử dụng điều kiện (class_idx)

    # Sinh ảnh từ latent vector
    img = G(latent_vector, label, truncation_psi=1.0, noise_mode='const')
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)  # Đảm bảo giá trị pixel hợp lệ

    # Lưu ảnh con vào thư mục đã chỉ định
    child_image_path = f"{output_dir}/generated_image_{age}.png"
    Image.fromarray(img[0].cpu().numpy(), 'RGB').save(child_image_path)  # Chỉnh sửa dòng này để sử dụng PIL.Image

    # Trả về đường dẫn của ảnh đã tạo
    return child_image_path

if __name__ == "__main__":
    generate_child_image("father_image.jpg", "mother_image.jpg", 3)  # Test for a sample generation