import os
import io
import threading
import logging
import traceback
import requests
import time
from tkinter import (
    Tk, Frame, Label, Button, Radiobutton,
    StringVar, filedialog, messagebox, OptionMenu, Text, END
)
from PIL import Image, ImageTk, ImageOps
from api_client import save_image_to_folder, upload_image_to_git, get_image_link_from_git, generate_baby_url  # Import function from api_client

# Logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')

def create_gui():
    root = Tk()
    root.title('Baby Face Generator')
    root.geometry('600x700')
    root.resizable(False, False)

    # Variables to hold file paths and selected options
    father_path = StringVar()
    mother_path = StringVar()
    gender = StringVar(value='random')
    age_choice = StringVar(value='baby')  # Default age to 'baby'

    # --- Upload & Preview Father/Mother Images ---
    pf = Frame(root)
    pf.pack(pady=20)
    fb = Frame(pf)
    fb.pack(side='left', padx=30)
    Label(fb, text='Father').pack()
    father_preview = Label(fb, text='Click to upload', width=20, height=10, bg='#eee')
    father_preview.pack()
    father_preview.bind("<Button-1>", lambda e: select_image(father_path, father_preview))

    mb = Frame(pf)
    mb.pack(side='left', padx=30)
    Label(mb, text='Mother').pack()
    mother_preview = Label(mb, text='Click to upload', width=20, height=10, bg='#eee')
    mother_preview.pack()
    mother_preview.bind("<Button-1>", lambda e: select_image(mother_path, mother_preview))

    # --- Gender radio buttons ---
    Label(root, text='Baby gender', font=('Arial', 12, 'bold')).pack(pady=(20, 5))
    gf = Frame(root)
    gf.pack()
    for txt, val in [('Random', 'random'), ('Boy', 'babyBoy'), ('Girl', 'babyGirl')]:
        rb = Radiobutton(gf, text=txt, variable=gender, value=val, indicatoron=0, width=10, padx=5, pady=5)
        rb.pack(side='left', padx=5)
        if val == 'random': rb.select()

    # --- Generate button ---
    Button(root, text='Generate Baby', font=('Arial', 14), bg='#3366FF', fg='white', width=30, height=2, command=lambda: threading.Thread(target=generate).start()).pack(pady=30)

    # --- Status Text Box ---
    result_box = Text(root, height=2, width=50)
    result_box.pack(pady=10)

    # --- Display Image Label ---
    image_label = Label(root)
    image_label.pack(pady=10)

    def select_image(var, preview):
        path = filedialog.askopenfilename(filetypes=[('Image', '*.png;*.jpg;*.jpeg')])
        if not path: return
        var.set(path)
        img = Image.open(path)

        # Resize ảnh để vừa với khung vuông tỉ lệ 1:1
        img = ImageOps.fit(img, (preview.winfo_width(), preview.winfo_width()), method=0, bleed=0.0, centering=(0.5, 0.5))

        # Convert ảnh thành PhotoImage để hiển thị trên preview
        ph = ImageTk.PhotoImage(img)
        preview.config(image=ph, text='')  # Đặt ảnh cho preview
        preview.image = ph  # Lưu ảnh vào preview để tránh mất ảnh khi giao diện thay đổi

    def generate():
        try:
            # Hiển thị trạng thái "Generating..."
            result_box.delete('1.0', END)
            result_box.insert(END, "Generating... Please wait...")

            if not father_path.get() or not mother_path.get():
                return messagebox.showerror('Error', 'Please select both parent images.')

            # Lấy giá trị giới tính từ giao diện
            gender_choice = gender.get()
            logging.debug(f"Selected Gender: {gender_choice}")

            # Lấy tên ảnh và lưu vào thư mục 'images'
            father_image_path_saved = save_image_to_folder(father_path.get(), True)
            mother_image_path_saved = save_image_to_folder(mother_path.get(), False)

            # Đẩy ảnh lên GitHub
            upload_image_to_git(father_image_path_saved)
            upload_image_to_git(mother_image_path_saved)

            # Lấy link ảnh từ GitHub
            father_img_url = get_image_link_from_git(os.path.basename(father_image_path_saved))
            mother_img_url = get_image_link_from_git(os.path.basename(mother_image_path_saved))

            # Gọi API Baby Generator để tạo ảnh em bé
            baby_bytes = generate_baby_url(father_img_url, mother_img_url, gender_choice)

            # Chuyển ảnh em bé thành ImageTk.PhotoImage và hiển thị
            img = Image.open(io.BytesIO(baby_bytes)).resize((400, 400), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            image_label.config(image=photo)
            image_label.image = photo

            # Hiển thị dòng trạng thái "Generation complete"
            result_box.delete('1.0', END)
            result_box.insert(END, "Generation complete!")

        except requests.exceptions.HTTPError as e:
            code, txt = e.response.status_code, e.response.text
            messagebox.showerror('HTTP Error', f"Status: {code}\n{txt}")
        except Exception as e:
            logging.error("Error in generate(): %s", e)
            traceback.print_exc()
            messagebox.showerror('Error', str(e))

    root.mainloop()

if __name__ == '__main__':
    create_gui()
