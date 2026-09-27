import numpy as np
import os, cv2, time, sys, glob
import torch, torchvision, argparse, model
import tkinter as tk
import tkinter.messagebox
import tkinter.font as font
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
from torchvision import transforms

# ── Base directory (same folder as this script) ──────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_INPUT_DIR  = os.path.join(BASE_DIR, "data", "test_data", "images")
RESULT_DIR      = os.path.join(BASE_DIR, "data", "result", "images")
SNAPSHOT_PATH   = os.path.join(BASE_DIR, "snapshots", "Epoch99.pth")
OUTPUT_PATH     = os.path.join(RESULT_DIR, "output.jpg")
INDEX_IMG_PATH  = os.path.join(BASE_DIR, "index.jpg")
ICON_IMG_PATH   = os.path.join(BASE_DIR, "download.png")

os.makedirs(TEST_INPUT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

originalImage = None

def upload():
    global originalImage, tkImage
    imagePath = filedialog.askopenfilename(
        title="Select a low-light image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not imagePath:
        return
    originalImage = cv2.imread(imagePath)
    originalImage = cv2.resize(originalImage, (600, 400))
    originalImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
    tkImage = Image.fromarray(originalImage)
    tkImage = ImageTk.PhotoImage(tkImage)
    Label(right_frame, image=tkImage).grid(row=0, column=0, padx=3, pady=3)

    IMAGE = cv2.imread(imagePath)
    cv2.imwrite(os.path.join(TEST_INPUT_DIR, "Inputimage.jpg"), IMAGE)
    tk.messagebox.showinfo("Upload", "Image uploaded successfully!")


def lowlight(image_path):
    global originalImage2, tkImage2
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_lowlight = Image.open(image_path)
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.unsqueeze(0)

    DCE_net = model.enhance_net_nopool()

    if not os.path.exists(SNAPSHOT_PATH):
        tk.messagebox.showerror("Model Not Found",
            f"Trained model not found at:\n{SNAPSHOT_PATH}\n\nPlease run lowlight_train.py first.")
        return

    DCE_net.load_state_dict(torch.load(SNAPSHOT_PATH, map_location='cpu'))
    DCE_net.eval()

    start = time.time()
    _, enhanced_image, _ = DCE_net(data_lowlight)
    print(f"Enhancement time: {time.time() - start:.3f}s")

    torchvision.utils.save_image(enhanced_image, OUTPUT_PATH)

    originalImage2 = cv2.imread(OUTPUT_PATH)
    originalImage2 = cv2.resize(originalImage2, (600, 400))
    originalImage2 = cv2.cvtColor(originalImage2, cv2.COLOR_BGR2RGB)
    tkImage2 = Image.fromarray(originalImage2)
    tkImage2 = ImageTk.PhotoImage(tkImage2)
    Label(mid_frame, image=tkImage2).grid(row=0, column=0, padx=3, pady=3)


def image_enhancement():
    with torch.no_grad():
        test_list = glob.glob(os.path.join(TEST_INPUT_DIR, "*"))
        if not test_list:
            tk.messagebox.showwarning("No Images",
                "No images found. Please upload an image first.")
            return
        for image in test_list:
            print(image)
            lowlight(image)
    tk.messagebox.showinfo("Done", "Image enhancement complete!")


def savefile():
    if not os.path.exists(OUTPUT_PATH):
        tk.messagebox.showerror("No Output",
            "No enhanced image found. Please enhance an image first.")
        return
    originalImage3 = cv2.imread(OUTPUT_PATH)
    originalImage3 = cv2.cvtColor(originalImage3, cv2.COLOR_BGR2RGB)
    edge = Image.fromarray(originalImage3)
    filename = filedialog.asksaveasfilename(
        defaultextension=".jpg",
        filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
    )
    if not filename:
        return
    edge.save(filename)
    tk.messagebox.showinfo("Saved", f"Image saved to:\n{filename}")


def Close():
    top.destroy()


# ── Build GUI ─────────────────────────────────────────────────────────────────
top = tk.Tk()
top.title('Image Enhancement using Deep Learning')
top.config(bg="grey16")

window_width, window_height = 1600, 700
screen_width  = top.winfo_screenwidth()
screen_height = top.winfo_screenheight()
center_x = int(screen_width / 2 - window_width / 2)
center_y = int(screen_height / 2 - window_height / 2)
top.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")

if os.path.exists(ICON_IMG_PATH):
    icon_img = ImageTk.PhotoImage(Image.open(ICON_IMG_PATH))
    top.iconphoto(False, icon_img)

left_frame  = Frame(top, width=300, height=530, bg='grey16')
left_frame.grid(row=1, column=0, padx=10, pady=55)
right_frame = Frame(top, width=700, height=530, bg='light slate grey')
right_frame.grid(row=1, column=1, padx=15, pady=25)
mid_frame   = Frame(top, width=700, height=530, bg='light slate grey')
mid_frame.grid(row=1, column=2, padx=15, pady=25)

# Placeholder images
if os.path.exists(INDEX_IMG_PATH):
    ai_image = ImageTk.PhotoImage(Image.open(INDEX_IMG_PATH).resize((600, 400)))
    Label(right_frame, image=ai_image).grid(row=0, column=0, padx=3, pady=3)
    a_image = ImageTk.PhotoImage(Image.open(INDEX_IMG_PATH).resize((600, 400)))
    Label(mid_frame, image=a_image).grid(row=0, column=0, padx=3, pady=3)

# Title
title_label = Label(top, text="LOW LIGHT IMAGE ENHANCEMENT",
                    font=("Times", 16, "bold"), bg="grey16", fg="white")
title_label.grid(row=0, column=0, columnspan=3, pady=10)

button_font = font.Font(family='Times', size=11)

upload_button = tk.Button(left_frame, text="UPLOAD IMAGE", command=upload,
                          fg="black", height=2, width=20)
upload_button.grid(row=0, column=1, padx=10, pady=15)
upload_button['font'] = button_font

enhance_button = tk.Button(left_frame, text="ENHANCE IMAGE", command=image_enhancement,
                           fg="black", height=2, width=20)
enhance_button.grid(row=1, column=1, padx=0, pady=15)
enhance_button['font'] = button_font

save_button = tk.Button(left_frame, text="SAVE IMAGE", command=savefile,
                        fg="black", height=2, width=20)
save_button.grid(row=4, column=1, padx=0, pady=15)
save_button['font'] = button_font

exit_button = tk.Button(left_frame, text="EXIT", command=Close,
                        fg="black", height=2, width=20)
exit_button.grid(row=5, column=1, padx=0, pady=15)
exit_button['font'] = button_font

Label(top, text="INPUT IMAGE", font=("Times", 12, "bold"),
      bg="grey16", fg="white").grid(row=2, column=1)
Label(top, text="ENHANCED IMAGE", font=("Times", 12, "bold"),
      bg="grey16", fg="white").grid(row=2, column=2)

top.mainloop()
