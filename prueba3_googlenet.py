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
class_names = ['R5', 'R6', 'R7', 'R8', 'R9', 'V1', 'V2', 'V3', 'V4']
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

model = models.googlenet(pretrained=False, aux_logits=False)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load('G19.pth', map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Interfaz Tkinter ---
BASE_FONT_SIZE = 18 # Aumentado el tamaño base de la fuente
INITIAL_SCREEN_WIDTH = 800
INITIAL_SCREEN_HEIGHT = 600 # Aumentado un poco para más espacio vertical

root = tk.Tk()
root.title('Clasificador de 9 Clases (Adaptable)')
root.geometry(f"{INITIAL_SCREEN_WIDTH}x{INITIAL_SCREEN_HEIGHT}")
root.minsize(600, 400) # Tamaño mínimo para que la UI no se rompa demasiado

# Grid principal (root) se expande
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

main_frame = tk.Frame(root, padx=4, pady=4)
main_frame.grid(row=0, column=0, sticky="nsew")
# Columnas del main_frame se reparten el espacio (3:2)
main_frame.grid_columnconfigure(0, weight=3) # Frame izquierda (imagen)
main_frame.grid_columnconfigure(1, weight=2) # Frame derecha (controles)
# Fila del main_frame se expande
main_frame.grid_rowconfigure(0, weight=1)

# Frame izquierda para imagen
left_frame = tk.Frame(main_frame, bd=2, relief=tk.SOLID)
left_frame.grid(row=0, column=0, sticky="nsew", padx=(0,4))
left_frame.grid_rowconfigure(0, weight=1)
left_frame.grid_columnconfigure(0, weight=1)

image_label = tk.Label(left_frame, text='Imagen', fg='gray', font=font.Font(size=BASE_FONT_SIZE+6)) # Fuente más grande para el placeholder
image_label.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

# Frame derecha para controles
right_frame = tk.Frame(main_frame, bd=2, relief=tk.SOLID)
right_frame.grid(row=0, column=1, sticky="nsew", padx=(4,0))
# Fila para status_label se expande
right_frame.grid_rowconfigure(0, weight=1)
# Fila para botones no se expande verticalmente (peso 0 es el default, pero lo explicitamos)
right_frame.grid_rowconfigure(1, weight=0)
# Columna única del right_frame se expande
right_frame.grid_columnconfigure(0, weight=1)

status_label = tk.Label(right_frame, text='Esperando acción...', font=font.Font(size=BASE_FONT_SIZE), justify=tk.LEFT, anchor='nw')
status_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

# Botones
button_frame = tk.Frame(right_frame)
button_frame.grid(row=1, column=0, sticky="sew", padx=5, pady=5) # "s" para pegar abajo
button_frame.grid_columnconfigure(0, weight=1) # La columna de botones se expande horizontalmente

capture_btn_font = font.Font(size=BASE_FONT_SIZE)
clear_btn_font = font.Font(size=BASE_FONT_SIZE)

capture_btn = tk.Button(button_frame, text='Tomar Foto', font=capture_btn_font, command=lambda: tomar_y_clasificar(), height=2)
capture_btn.grid(row=0, column=0, sticky="ew", pady=(0,5))

clear_btn = tk.Button(button_frame, text='Limpiar', font=clear_btn_font, command=lambda: limpiar(), state=tk.DISABLED, height=2)
clear_btn.grid(row=1, column=0, sticky="ew")

last_photo_path = None
current_pil_image = None # Para guardar la imagen PIL original para redimensionar
resize_job = None # Para el debounce del redimensionamiento de imagen

def classify_image(path_or_pil_image):
    if isinstance(path_or_pil_image, str):
        img = Image.open(path_or_pil_image).convert('RGB')
    elif isinstance(path_or_pil_image, Image.Image):
        img = path_or_pil_image.convert('RGB')
    else:
        raise ValueError("Se espera una ruta de archivo o un objeto PIL.Image")

    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        _, pred = torch.max(out,1)
    return class_names[pred.item()]

def display_image(pil_img_to_display):
    """Muestra una imagen PIL en image_label, redimensionándola."""
    global current_pil_image
    current_pil_image = pil_img_to_display # Guardar para redimensionado posterior

    # Esperar a que Tkinter actualice las dimensiones del widget
    image_label.update_idletasks()
    max_w = image_label.winfo_width() - 10
    max_h = image_label.winfo_height() - 10

    if max_w <= 10 or max_h <= 10: # Si el widget es muy pequeño o no está listo
        # Usar un fallback o esperar. Por ahora, no mostramos si es muy pequeño.
        # Podrías también poner un tamaño mínimo como 100x100
        img_copy = current_pil_image.copy()
        img_copy.thumbnail((INITIAL_SCREEN_WIDTH * 0.5, INITIAL_SCREEN_HEIGHT * 0.5), Image.LANCZOS) # Fallback a tamaño grande
    else:
        img_copy = current_pil_image.copy()
        img_copy.thumbnail((max_w, max_h), Image.LANCZOS)

    tk_img = ImageTk.PhotoImage(img_copy)
    image_label.config(image=tk_img, text='')
    image_label.image = tk_img


def tomar_y_clasificar():
    global last_photo_path, current_pil_image
    status_label.config(text='Capturando imagen...')
    root.update_idletasks()

    save_dir = 'fotos'
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ruta = os.path.join(save_dir, f'captura_{timestamp}.jpg')
    try:
        subprocess.run(['libcamera-jpeg','-n','-o',ruta,'-t','200', '--width', '1280', '--height', '960'], check=True)
    except FileNotFoundError:
        status_label.config(text='Error: libcamera-jpeg no encontrado.')
        return
    except subprocess.CalledProcessError as e:
        status_label.config(text=f'Error al capturar: {e}')
        return
    except Exception as e:
        status_label.config(text=f'Error captura general: {e}')
        return

    try:
        pil_image = Image.open(ruta)
        display_image(pil_image) # Llama a la nueva función para mostrar
    except Exception as e:
        status_label.config(text=f'Error al mostrar imagen: {e}')
        return

    status_label.config(text='Clasificando...')
    root.update_idletasks()
    try:
        predicted_class_name = classify_image(pil_image) # Clasificar desde la imagen PIL en memoria
        description = class_descriptions.get(predicted_class_name, f"Descripción no encontrada para: {predicted_class_name}")
        status_label.config(text=description)
    except Exception as e:
        status_label.config(text=f'Error al clasificar: {e}')
        return

    capture_btn.config(state=tk.DISABLED)
    clear_btn.config(state=tk.NORMAL)
    last_photo_path = ruta

def limpiar():
    global last_photo_path, current_pil_image
    if last_photo_path and os.path.exists(last_photo_path):
        try: os.remove(last_photo_path)
        except Exception as e: print(f"No se pudo borrar {last_photo_path}: {e}")
    last_photo_path = None
    current_pil_image = None
    image_label.config(image=None, text='Imagen', fg='gray', font=font.Font(size=BASE_FONT_SIZE+6)) # Restaurar fuente placeholder
    image_label.image = None
    status_label.config(text='Esperando acción...')
    capture_btn.config(state=tk.NORMAL)
    clear_btn.config(state=tk.DISABLED)

def update_status_wraplength(event=None):
    """Ajusta el wraplength del status_label al ancho del right_frame."""
    # Damos un pequeño margen para el padding interno del label o del frame
    new_width = right_frame.winfo_width() - 20
    if new_width > 0 : # Asegurarse que el ancho es positivo
        status_label.config(wraplength=new_width)

def on_image_label_configure(event):
    """Se llama cuando image_label cambia de tamaño. Redibuja la imagen con debounce."""
    global resize_job, current_pil_image
    if current_pil_image:
        if resize_job:
            root.after_cancel(resize_job)
        # Espera 250ms después del último evento de redimensionamiento antes de actuar
        resize_job = root.after(250, lambda: actual_image_resize_on_configure())

def actual_image_resize_on_configure():
    """Realiza el redimensionamiento de la imagen."""
    global current_pil_image
    if not current_pil_image or not image_label.winfo_exists(): # Chequear si el widget aún existe
        return

    max_w = image_label.winfo_width() - 10
    max_h = image_label.winfo_height() - 10

    if max_w <= 10 or max_h <= 10: # Evitar errores si el widget es demasiado pequeño
        return

    img_copy = current_pil_image.copy()
    img_copy.thumbnail((max_w, max_h), Image.LANCZOS)
    tk_img = ImageTk.PhotoImage(img_copy)
    image_label.config(image=tk_img)
    image_label.image = tk_img


# Bindings para responsividad
right_frame.bind("<Configure>", update_status_wraplength)
image_label.bind("<Configure>", on_image_label_configure)


# Llamada inicial para configurar wraplength, por si acaso
root.update_idletasks() # Asegura que los widgets tengan dimensiones iniciales
update_status_wraplength()


root.mainloop()