import os
import subprocess
import logging
import requests
from PIL import Image
import time
import base64

# === MaxStudio API config ===
API_KEY      = '19034319-1689-43df-b1d7-ef56efe394bb'
BABY_GEN_URL = 'https://api.maxstudio.ai/baby-generator'
POST_HEADERS = {
    'x-api-key': API_KEY,
    'Content-Type': 'application/json'
}
GET_HEADERS = {
    'x-api-key': API_KEY
}

# === Image folder config ===
IMAGES_DIR = "images"
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

RESULTS_DIR = "results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')

def get_unique_file_name(file_path: str) -> str:
    """
    Kiểm tra xem file đã tồn tại chưa. Nếu có thì thêm hậu tố vào tên file để đảm bảo không trùng lặp.
    """
    base_name, ext = os.path.splitext(file_path)
    counter = 1
    new_file_path = file_path
    while os.path.exists(new_file_path):
        new_file_path = f"{base_name}_{counter}{ext}"
        counter += 1
    return new_file_path

def save_image_to_folder(image_path: str, is_father: bool) -> str:
    """
    Lưu ảnh vào thư mục images với tên file dựa trên loại ảnh (father/mother).
    Tránh việc trùng tên bằng cách thêm hậu tố vào tên tệp nếu cần.
    """
    # Mở ảnh và kiểm tra nếu cần chuyển đổi định dạng
    img = Image.open(image_path)

    # Nếu ảnh có kênh alpha (RGBA), convert sang RGB trước khi lưu JPEG
    if img.mode == "RGBA":
        img = img.convert("RGB")

    # Lấy tên file gốc từ đường dẫn ảnh
    base_name = os.path.basename(image_path)  # Lấy tên file từ đường dẫn
    name_without_ext, ext = os.path.splitext(base_name)  # Tách phần tên và phần mở rộng của file

    # Đặt tên mới cho ảnh
    suffix = "_father" if is_father else "_mother"
    file_name = f"{name_without_ext}{suffix}{ext}"

    # Đảm bảo tên file duy nhất
    file_path = os.path.join(IMAGES_DIR, file_name)
    file_path = get_unique_file_name(file_path)  # Kiểm tra trùng tên và sửa lại nếu cần

    # Lưu ảnh vào thư mục 'images'
    img.save(file_path)

    logging.debug(f"Image saved to {file_path}")
    return file_path

def upload_image_to_git(image_path: str):
    """
    Đẩy ảnh vừa được lưu vào GitHub.
    """
    try:
        # Kiểm tra xem file có tồn tại không
        if not os.path.exists(image_path):
            raise RuntimeError(f"File {image_path} does not exist.")

        # Thêm file vào Git repository
        subprocess.run(["git", "add", image_path], check=True)

        # Commit các thay đổi với thông điệp
        subprocess.run(["git", "commit", "-m", f"Add {image_path}"], check=True)

        # Đẩy thay đổi lên GitHub
        subprocess.run(["git", "push", "-f"], check=True)
        
        logging.debug(f"Uploaded {image_path} to GitHub.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Git command failed: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise

def get_image_link_from_git(image_name: str) -> str:
    """
    Lấy link ảnh từ GitHub.
    """
    return f"https://github.com/xuanloc251199/future_baby_face_generator_ai/raw/main/{IMAGES_DIR}/{image_name}"

def _poll_job(job_id: str, url: str) -> dict:
    """
    Poll trạng thái job MaxStudio đến khi completed.
    """
    url = f"{url}/{job_id}"
    logging.debug(f"Start polling job: {url}")
    
    while True:
        try:
            r = requests.get(url, headers=GET_HEADERS)
            logging.debug(f"Poll {url} → {r.status_code}")
            if r.status_code == 200:
                data = r.json()
                status = data.get('status')
                logging.debug(f"Job {job_id} status: {status}")
                if status == 'completed':
                    return data
                if status in ('failed', 'not-found'):
                    raise RuntimeError(f"Job {job_id} failed: {status}")
            else:
                logging.warning(f"Unexpected {r.status_code} response. Retrying...")
        except requests.exceptions.RequestException as e:
            logging.error(f"Error during poll {url}: {e}")
        # Keep polling without delay

def generate_baby_url(father_url: str, mother_url: str, gender: str) -> bytes:
    """
    Tạo ảnh em bé từ ảnh của bố và mẹ.
    """
    payload = {
        'fatherImage': father_url,
        'motherImage': mother_url,
        'gender': gender
    }
    logging.debug(f"POST {BABY_GEN_URL} payload={payload}")
    r = requests.post(BABY_GEN_URL, headers=POST_HEADERS, json=payload)
    logging.debug(f"Response: {r.status_code} {r.text}")
    r.raise_for_status()

    job_id = r.json().get('jobId')
    logging.debug(f"Job ID: {job_id}")

    result = _poll_job(job_id, BABY_GEN_URL)
    baby_url = result['result'][0]
    logging.debug(f"Baby image URL: {baby_url}")

    img_resp = requests.get(baby_url)
    logging.debug(f"Downloaded baby image: {img_resp.status_code}, {len(img_resp.content)} bytes")
    img_resp.raise_for_status()
    
    # Save image to 'results' folder
    image_path = os.path.join(RESULTS_DIR, f"generated_baby_{int(time.time())}.jpg")
    with open(image_path, 'wb') as f:
        f.write(img_resp.content)
    
    return img_resp.content
