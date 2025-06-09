import tkinter as tk
import tkinter.font as tkfont
from PIL import Image, ImageTk
import os
from datetime import datetime
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

model = models.googlenet(weights=None, aux_logits=False)
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
BASE_FONT_SIZE   = 20-8    # Tamaño base de fuente para textos generales
BUTTON_FONT_SIZE = 18-8    # Tamaño de fuente para los botones
STAGE_FONT_SIZE  = 24-8    # Tamaño de fuente para el subtítulo "Etapa: ..."
# Ahora la descripción usa la misma fuente que los botones:
DESC_FONT_SIZE   = BUTTON_FONT_SIZE   # Tamaño de fuente para la descripción (igual que botones)
BUTTON_FONT_FAM  = "Arial"        # Familia de fuente para botones
STAGE_FONT_FAM   = "Helvetica"    # Familia de fuente para el subtítulo
DESC_FONT_FAM    = BUTTON_FONT_FAM   # Familia de fuente para la descripción (igual que botones)
IMAGE_FONT_FAM   = "Helvetica"    # Fuente del placeholder de "Imagen"

INITIAL_SCREEN_WIDTH  = 320
INITIAL_SCREEN_HEIGHT = 480

root = tk.Tk()
root.title('Clasificador de Etapas de Frijol')
root.attributes('-zoomed', True)

# Comentamos el minsize grande porque no sirve en pantallas pequeñas:
# root.minsize(700, 500)

# --- Ajuste global de fuentes nombradas para textos generales (Etiquetas, Menús, etc.) ---
default_font = tkfont.nametofont("TkDefaultFont")
default_font.configure(size=BASE_FONT_SIZE, family="Sans")
text_font    = tkfont.nametofont("TkTextFont")
text_font.configure(size=BASE_FONT_SIZE, family="Sans")
menu_font    = tkfont.nametofont("TkMenuFont")
menu_font.configure(size=BASE_FONT_SIZE, family="Sans")

# Grid principal (root) se expande
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

# main_frame con márgenes reducidos
main_frame = tk.Frame(root, padx=4, pady=4)
main_frame.grid(row=0, column=0, sticky="nsew")
main_frame.grid_columnconfigure(0, weight=3)  # Frame izquierda (imagen)
main_frame.grid_columnconfigure(1, weight=2)  # Frame derecha (controles)
main_frame.grid_rowconfigure(0, weight=1)

# Frame izquierda para imagen (borde más fino)
left_frame = tk.Frame(main_frame, bd=1, relief=tk.SOLID)
left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
left_frame.grid_rowconfigure(0, weight=1)
left_frame.grid_columnconfigure(0, weight=1)

# Placeholder “Imagen” con padding reducido
image_label = tk.Label(
    left_frame,
    text='Imagen',
    fg='gray',
    font=tkfont.Font(family=IMAGE_FONT_FAM, size=BASE_FONT_SIZE + 4, weight='bold')
)
image_label.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

# Frame derecha para controles (borde más fino)
right_frame = tk.Frame(main_frame, bd=1, relief=tk.SOLID)
right_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
right_frame.grid_rowconfigure(0, weight=0)  # Subtítulo “Etapa”
right_frame.grid_rowconfigure(1, weight=1)  # Descripción
right_frame.grid_rowconfigure(2, weight=0)  # Botones
right_frame.grid_columnconfigure(0, weight=1)

# Subtítulo “Etapa: ...” con padding reducido
stage_label = tk.Label(
    right_frame,
    text='',
    fg='blue',
    font=tkfont.Font(family=STAGE_FONT_FAM, size=STAGE_FONT_SIZE, weight='bold'),
    anchor='nw',
    justify=tk.LEFT
)
stage_label.grid(row=0, column=0, sticky="nw", padx=8, pady=(8, 4))

# Etiqueta para mostrar la descripción con wraplength ajustado luego
status_label = tk.Label(
    right_frame,
    text='Esperando acción...',
    font=tkfont.Font(family=DESC_FONT_FAM, size=DESC_FONT_SIZE),
    anchor='nw',
    justify=tk.LEFT
)
status_label.grid(row=1, column=0, sticky="nsew", padx=8, pady=(4, 8))
status_label.config(wraplength=(INITIAL_SCREEN_WIDTH * 2 // 5) - 10)

# Botones (sin height=2 para no ocupar tanto vertical)
button_frame = tk.Frame(right_frame)
button_frame.grid(row=2, column=0, sticky="sew", padx=5, pady=5)
button_frame.grid_columnconfigure(0, weight=1)

# “Tomar Foto”
capture_btn = tk.Button(
    button_frame,
    text='Tomar Foto',
    font=tkfont.Font(family=BUTTON_FONT_FAM, size=BUTTON_FONT_SIZE, weight='bold'),
    command=lambda: tomar_y_clasificar()
)
capture_btn.grid(row=0, column=0, sticky="ew", pady=(0, 4))

# “Limpiar”
clear_btn = tk.Button(
    button_frame,
    text='Limpiar',
    font=tkfont.Font(family=BUTTON_FONT_FAM, size=BUTTON_FONT_SIZE),
    command=lambda: limpiar(),
    state=tk.DISABLED
)
clear_btn.grid(row=1, column=0, sticky="ew")

last_photo_path = None
current_pil_image = None  # Para guardar la imagen PIL original
resize_job = None         # Para el debounce del redimensionamiento

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
        _, pred = torch.max(out, 1)
    return class_names[pred.item()]

def display_image(pil_img_to_display):
    """Muestra una imagen PIL en image_label, rotada y redimensionada."""
    # 1. Rotar 90° a la derecha
    pil_img_to_display = pil_img_to_display.rotate(-90, expand=True)

    global current_pil_image
    current_pil_image = pil_img_to_display  # Guardar para redimensionado posterior

    image_label.update_idletasks()
    max_w = image_label.winfo_width() - 10
    max_h = image_label.winfo_height() - 10

    if max_w <= 10 or max_h <= 10:
        img_copy = current_pil_image.copy()
        img_copy.thumbnail((INITIAL_SCREEN_WIDTH * 0.5, INITIAL_SCREEN_HEIGHT * 0.5), Image.LANCZOS)
    else:
        img_copy = current_pil_image.copy()
        img_copy.thumbnail((max_w, max_h), Image.LANCZOS)

    tk_img = ImageTk.PhotoImage(img_copy)
    image_label.config(image=tk_img, text='')
    image_label.image = tk_img

def tomar_y_clasificar():
    global last_photo_path, current_pil_image
    stage_label.config(text='')
    status_label.config(text='Capturando imagen...', font=tkfont.Font(family=DESC_FONT_FAM, size=DESC_FONT_SIZE))
    root.update_idletasks()

    save_dir = 'fotos_campo'
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ruta = os.path.join(save_dir, f'captura_{timestamp}.jpg')
    try:
        subprocess.run(
            ['libcamera-jpeg', '-n', '-o', ruta, '-t', '200', '--width', '1280', '--height', '960'],
            check=True
        )
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
        display_image(pil_image)
    except Exception as e:
        status_label.config(text=f'Error al mostrar imagen: {e}')
        return

    status_label.config(text='Clasificando...', font=tkfont.Font(family=DESC_FONT_FAM, size=DESC_FONT_SIZE))
    root.update_idletasks()
    try:
        predicted_class_name = classify_image(pil_image)
        description = class_descriptions.get(
            predicted_class_name,
            f"Descripción no encontrada para: {predicted_class_name}"
        )
        # Mostrar la etapa clasificada
        stage_label.config(text=f"Etapa: {predicted_class_name}")
        status_label.config(text=description)
    except Exception as e:
        status_label.config(text=f'Error al clasificar: {e}')
        return

    capture_btn.config(state=tk.DISABLED)
    clear_btn.config(state=tk.NORMAL)
    last_photo_path = ruta

def limpiar():
    global last_photo_path, current_pil_image
    # if last_photo_path and os.path.exists(last_photo_path):
    #     try:
    #         os.remove(last_photo_path)
    #     except Exception as e:
    #         print(f"No se pudo borrar {last_photo_path}: {e}")
    last_photo_path = None
    current_pil_image = None

    image_label.config(
        image=None,
        text='Imagen',
        fg='gray',
        font=tkfont.Font(family=IMAGE_FONT_FAM, size=BASE_FONT_SIZE + 4, weight='bold')
    )
    image_label.image = None
    stage_label.config(text='')
    status_label.config(
        text='Esperando acción...',
        font=tkfont.Font(family=DESC_FONT_FAM, size=DESC_FONT_SIZE)
    )
    capture_btn.config(state=tk.NORMAL)
    clear_btn.config(state=tk.DISABLED)

def update_status_wraplength(event=None):
    """Ajusta el wraplength del status_label al ancho del right_frame."""
    new_width = right_frame.winfo_width() - 30
    if new_width > 0:
        status_label.config(wraplength=new_width)

def on_image_label_configure(event):
    """Se llama cuando image_label cambia de tamaño. Redibuja la imagen con debounce."""
    global resize_job, current_pil_image
    if current_pil_image:
        if resize_job:
            root.after_cancel(resize_job)
        resize_job = root.after(250, actual_image_resize_on_configure)

def actual_image_resize_on_configure():
    """Realiza el redimensionamiento de la imagen."""
    global current_pil_image
    if not current_pil_image or not image_label.winfo_exists():
        return

    max_w = image_label.winfo_width() - 10
    max_h = image_label.winfo_height() - 10

    if max_w <= 10 or max_h <= 10:
        return

    img_copy = current_pil_image.copy()
    img_copy.thumbnail((max_w, max_h), Image.LANCZOS)
    tk_img = ImageTk.PhotoImage(img_copy)
    image_label.config(image=tk_img)
    image_label.image = tk_img

# Bindings para responsividad
right_frame.bind("<Configure>", update_status_wraplength)
image_label.bind("<Configure>", on_image_label_configure)

# Llamada inicial para configurar wraplength
root.update_idletasks()
update_status_wraplength()

root.mainloop()