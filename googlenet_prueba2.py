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
# Nuevos nombres de clases
class_names = ['R5', 'R6', 'R7', 'R8', 'R9', 'V1', 'V2', 'V3', 'V4']

# Textos descriptivos para cada clase
class_descriptions = {
    'R5': "Inicio de la floración; se observan las primeras flores abiertas.",
    'R6': "Floración plena; la mayoría de las plantas tienen flores.",
    'R7': "Formación de vainas; las primeras vainas jóvenes son visibles.",
    'R8': "Llenado de vainas; las semillas dentro de las vainas comienzan a desarrollarse.",
    'R9': "Madurez fisiológica; las semillas alcanzan su tamaño y peso máximo, y las vainas comienzan a secarse.",
    'V1': "Emergencia del primer par de hojas unifoliadas completamente abiertas.",
    'V2': "Aparición del primer par de hojas trifoliadas completamente desarrolladas.",
    'V3': "Desarrollo del segundo par de hojas trifoliadas.",
    'V4': "Desarrollo del tercer par de hojas trifoliadas; la planta continúa acumulando biomasa"
}


model = models.googlenet(weights=None, aux_logits=False)  # Desactiva los clasificadores auxiliares
model.fc = nn.Linear(model.fc.in_features, len(class_names)) # len(class_names) sigue siendo 9
model.load_state_dict(torch.load('G19.pth', map_location=device)) # Asegúrate que G19.pth fue entrenado para 9 clases
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
BASE_FONT_SIZE = 14
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 480

root = tk.Tk()
root.title('Clasificador de 9 Clases (R5-R9, V1-V4)')
root.geometry(f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}")

# Grid principal
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

main_frame = tk.Frame(root, padx=4, pady=4)
main_frame.grid(row=0, column=0, sticky="nsew")
main_frame.grid_columnconfigure(0, weight=3)
main_frame.grid_columnconfigure(1, weight=2)
main_frame.grid_rowconfigure(0, weight=1)

# Frame izquierda para imagen
left_frame = tk.Frame(main_frame, bd=2, relief=tk.SOLID)
left_frame.grid(row=0, column=0, sticky="nsew", padx=(0,4))
left_frame.grid_rowconfigure(0, weight=1)
left_frame.grid_columnconfigure(0, weight=1)

image_label = tk.Label(left_frame, text='Imagen', fg='gray', font=font.Font(size=BASE_FONT_SIZE+4))
image_label.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

# Frame derecha para controles
right_frame = tk.Frame(main_frame, bd=2, relief=tk.SOLID)
right_frame.grid(row=0, column=1, sticky="nsew", padx=(4,0))
right_frame.grid_rowconfigure(0, weight=1) # Espacio para el texto de estado
right_frame.grid_rowconfigure(1, weight=0) # Espacio para botones (no expandir)
right_frame.grid_columnconfigure(0, weight=1)

status_label = tk.Label(right_frame, text='Esperando acción...', font=font.Font(size=BASE_FONT_SIZE), wraplength=(SCREEN_WIDTH*2//5)-30, justify=tk.LEFT, anchor='nw')
status_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10) # Aumentado padding

# Botones
button_frame = tk.Frame(right_frame)
button_frame.grid(row=1, column=0, sticky="sew", padx=5, pady=5) # Sticky "sew" para alinear abajo
button_frame.grid_columnconfigure(0, weight=1)

capture_btn = tk.Button(button_frame, text='Tomar Foto', font=font.Font(size=BASE_FONT_SIZE), command=lambda: tomar_y_clasificar(), height=2)
capture_btn.grid(row=0, column=0, sticky="ew", pady=(0,5))

clear_btn = tk.Button(button_frame, text='Limpiar', font=font.Font(size=BASE_FONT_SIZE), command=lambda: limpiar(), state=tk.DISABLED, height=2)
clear_btn.grid(row=1, column=0, sticky="ew")

root.update_idletasks()

last_photo_path = None

def classify_image(path):
    img = Image.open(path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        _, pred = torch.max(out,1)
    return class_names[pred.item()] # Devuelve el nombre de la clase, ej: 'R5'

def tomar_y_clasificar():
    global last_photo_path
    status_label.config(text='Capturando imagen...')
    root.update_idletasks()

    save_dir = 'fotos'
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ruta = os.path.join(save_dir, f'captura_{timestamp}.jpg')
    try:
        # Ajusta este comando si 'libcamera-jpeg' no está en tu PATH o si usas otra herramienta
        subprocess.run(['libcamera-jpeg','-n','-o',ruta,'-t','200', '--width', '640', '--height', '480'], check=True)
    except FileNotFoundError:
        status_label.config(text='Error: Comando libcamera-jpeg no encontrado. Asegúrate que está instalado y en el PATH.')
        return
    except subprocess.CalledProcessError as e:
        status_label.config(text=f'Error al capturar con libcamera: {e}')
        return
    except Exception as e:
        status_label.config(text=f'Error captura general: {e}')
        return

    # Mostrar imagen
    try:
        img = Image.open(ruta)
        # Determinar tamaño máximo basado en el espacio del label
        # Esperar a que Tkinter actualice las dimensiones del widget
        image_label.update_idletasks()
        max_w = image_label.winfo_width() - 10 # -10 para un pequeño margen
        max_h = image_label.winfo_height() - 10

        if max_w <= 0 or max_h <= 0: # Fallback si el widget aún no tiene tamaño
            max_w, max_h = 400, 300

        img.thumbnail((max_w, max_h), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        image_label.config(image=tk_img, text='')
        image_label.image = tk_img # Guardar referencia para evitar garbage collection
    except Exception as e:
        status_label.config(text=f'Error al mostrar imagen: {e}')
        return

    # Clasificar
    status_label.config(text='Clasificando...')
    root.update_idletasks()
    try:
        predicted_class_name = classify_image(ruta)
        # Obtener la descripción de la clase
        description = class_descriptions.get(predicted_class_name, f"Descripción no encontrada para la clase: {predicted_class_name}")
        status_label.config(text=description)
    except Exception as e:
        status_label.config(text=f'Error al clasificar: {e}')
        # No desactivar botones si la clasificación falla, para poder reintentar con otra foto.
        return


    capture_btn.config(state=tk.DISABLED)
    clear_btn.config(state=tk.NORMAL)
    last_photo_path = ruta

def limpiar():
    global last_photo_path
    if last_photo_path and os.path.exists(last_photo_path):
        try:
            os.remove(last_photo_path)
        except Exception as e:
            print(f"No se pudo borrar {last_photo_path}: {e}") # Log al console
            pass # Continuar aunque no se pueda borrar
    last_photo_path = None
    image_label.config(image=None, text='Imagen', fg='gray')
    image_label.image = None # Limpiar referencia
    status_label.config(text='Esperando acción...')
    capture_btn.config(state=tk.NORMAL)
    clear_btn.config(state=tk.DISABLED)

root.mainloop()