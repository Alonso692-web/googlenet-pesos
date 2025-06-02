import tkinter as tk
from tkinter import font
import os
from datetime import datetime
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import models, transforms
import subprocess

# --- Configuración modelo ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_names = ['clase0','clase1','clase2','clase3','clase4','clase5','clase6','clase7','clase8']
model = models.resnet34(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load('R23.pth', map_location=device))
model.to(device)
model.eval()

# --- Transformación de imagen ---
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Interfaz Tkinter ---
BASE_FONT_SIZE = 12
BUTTON_FONT_SIZE = 13
SCREEN_WIDTH = 320
SCREEN_HEIGHT = 480

root = tk.Tk()
root.title('Clasificador de 9 Clases')
root.geometry(f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}")
root.minsize(320, 480)

# Grid principal
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

main_frame = tk.Frame(root, padx=2, pady=2)
main_frame.grid(row=0, column=0, sticky="nsew")
main_frame.grid_columnconfigure(0, weight=3)
main_frame.grid_columnconfigure(1, weight=2)
main_frame.grid_rowconfigure(0, weight=1)

# Frame izquierda para imagen
left_frame = tk.Frame(main_frame, bd=1, relief=tk.SOLID)
left_frame.grid(row=0, column=0, sticky="nsew", padx=(0,2))
left_frame.grid_rowconfigure(0, weight=1)
left_frame.grid_columnconfigure(0, weight=1)

image_label = tk.Label(left_frame, text='Imagen', fg='gray', font=font.Font(size=BASE_FONT_SIZE+2))
image_label.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

# Frame derecha para controles
right_frame = tk.Frame(main_frame, bd=1, relief=tk.SOLID)
right_frame.grid(row=0, column=1, sticky="nsew", padx=(2,0))
right_frame.grid_rowconfigure(0, weight=1)
right_frame.grid_rowconfigure(1, weight=0)
right_frame.grid_columnconfigure(0, weight=1)

# Etiqueta de estado con wrap ajustado
status_label = tk.Label(
    right_frame,
    text='Esperando acción...',
    font=font.Font(size=BASE_FONT_SIZE),
    wraplength=(SCREEN_WIDTH * 2 // 5) - 10,
    justify=tk.LEFT,
    anchor='nw'
)
status_label.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

# Botones
button_frame = tk.Frame(right_frame)
button_frame.grid(row=1, column=0, sticky="ew", padx=4, pady=4)
button_frame.grid_columnconfigure(0, weight=1)

capture_btn = tk.Button(
    button_frame,
    text='Tomar Foto',
    font=font.Font(size=BUTTON_FONT_SIZE),
    command=lambda: tomar_y_clasificar()
)
capture_btn.grid(row=0, column=0, sticky="ew", pady=(0,4))

clear_btn = tk.Button(
    button_frame,
    text='Limpiar',
    font=font.Font(size=BUTTON_FONT_SIZE),
    command=lambda: limpiar(),
    state=tk.DISABLED
)
clear_btn.grid(row=1, column=0, sticky="ew")

root.update_idletasks()

last_photo_path = None

def classify_image(path):
    img = Image.open(path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        _, pred = torch.max(out,1)
    return class_names[pred.item()]

def tomar_y_clasificar():
    global last_photo_path
    status_label.config(text='Capturando imagen...')
    root.update_idletasks()

    save_dir = 'fotos'
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ruta = os.path.join(save_dir, f'captura_{timestamp}.jpg')
    try:
        subprocess.run(['libcamera-jpeg','-n','-o',ruta,'-t','200'], check=True)
    except Exception as e:
        status_label.config(text=f'Error captura: {e}')
        return

    try:
        img = Image.open(ruta)
        max_w = image_label.winfo_width() - 10 or 150
        max_h = image_label.winfo_height() - 10 or 150
        img.thumbnail((max_w, max_h), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        image_label.config(image=tk_img, text='')
        image_label.image = tk_img
    except Exception as e:
        status_label.config(text=f'Error mostrar: {e}')
        return

    res = classify_image(ruta)
    status_label.config(text=f'Predicción: {res}')
    capture_btn.config(state=tk.DISABLED)
    clear_btn.config(state=tk.NORMAL)
    last_photo_path = ruta

def limpiar():
    global last_photo_path
    if last_photo_path and os.path.exists(last_photo_path):
        try: os.remove(last_photo_path)
        except: pass
    last_photo_path = None
    image_label.config(image=None, text='Imagen', fg='gray')
    image_label.image = None
    status_label.config(text='Esperando acción...')
    capture_btn.config(state=tk.NORMAL)
    clear_btn.config(state=tk.DISABLED)

root.mainloop()