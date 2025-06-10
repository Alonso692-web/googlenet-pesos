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

# Aquí la nueva estructura con descripción y recomendaciones
class_recomendaciones = {
    'V1': {
        "descripcion": "Emergencia del primer par de hojas unifoliadas completamente abiertas.",
        "recomendaciones": (
            "Riego: Mantener humedad uniforme en el perfil de suelo para favorecer desarrollo radicular. "
            "Control de malezas: Paso temprano de escarda o herbicida residual selectivo. "
            "Salud de suelo: Revisar pH y enmendar con cal si sea necesario."
        )
    },
    'V2': {
        "descripcion": "Aparición del primer par de hojas trifoliadas completamente desarrolladas.",
        "recomendaciones": (
            "Fertilización: Primera dosis de nitrógeno (si no hay buena inoculación bacteriana), monitorear fósforo y potasio. "
            "Monitoreo de plagas: Inspeccionar áfidos y trips; usar trampas cromáticas. "
            "Control de malezas tardías: Segundo pase de escarda o herbicida residual."
        )
    },
    'V3': {
        "descripcion": "Desarrollo del segundo par de hojas trifoliadas.",
        "recomendaciones": (
            "Micronutrientes: Aplicación foliar ligera de zinc y boro si hay clorosis o necrosis. "
            "Salud foliar: Revisar síntomas de roya y manchas; fungicida preventivo si aparecen. "
            "Riego: Ajustar frecuencia según humedad del suelo para evitar estrés hídrico."
        )
    },
    'V4': {
        "descripcion": "Desarrollo del tercer par de hojas trifoliadas; la planta continúa acumulando biomasa.",
        "recomendaciones": (
            "Dosel y densidad: Evaluar cobertura del suelo y considerar ajustes de densidad para el próximo ciclo. "
            "Control biológico: Fomentar insectos benéficos con bandas florales o refugios. "
            "Nutrición: Refuerzo de potasio para mejorar resistencia al estrés."
        )
    },
    'R5': {
        "descripcion": "Inicio de la floración; se observan las primeras flores abiertas.",
        "recomendaciones": (
            "Riego crítico: Humedad constante durante floración para evitar caída de flores. "
            "Fungicidas: Aplicar mezcla de contacto + sistémico al inicio de floración. "
            "Insecticidas: Monitorear y controlar chinches de soya y mosquita blanca."
        )
    },
    'R6': {
        "descripcion": "Floración plena; la mayoría de las plantas tienen flores.",
        "recomendaciones": (
            "Monitoreo climático: Refuerzo de fungicida en periodos lluviosos y riego de auxilio en calor extremo. "
            "Foliar: Aporte de calcio y magnesio para mejorar cuajado de vainas. "
            "Control integrado: Mantener trampas y seguimiento de plagas."
        )
    },
    'R7': {
        "descripcion": "Formación de vainas; las primeras vainas jóvenes son visibles.",
        "recomendaciones": (
            "Fertilización de fondo: Si el análisis de tejido lo indica, aplicar fertilizante de liberación lenta. "
            "Riego de socorro: Mantener 60–70 % de capacidad de campo para proteger número de semillas. "
            "Inspección de frutos: Vigilar daños por trips y chinches."
        )
    },
    'R8': {
        "descripcion": "Llenado de vainas; las semillas dentro de las vainas comienzan a desarrollarse.",
        "recomendaciones": (
            "Riego óptimo: Evitar déficit hídrico, etapa crítica para rendimiento. "
            "Enfermedades de vainas: Revisar antracnosis y aplicar fungicida si hay manchas. "
            "Nutrición final: Aporte foliar de potasio para mejorar transporte de fotosintatos."
        )
    },
    'R9': {
        "descripcion": "Madurez fisiológica; las semillas alcanzan su tamaño y peso máximo, y las vainas comienzan a secarse.",
        "recomendaciones": (
            "Reducción de riego: Suspender cuando las vainas empiecen a secar para facilitar madurez. "
            "Cosecha: Vigilar humedad de grano (18–20 %) para programar fecha óptima. "
            "Prevención de pérdidas: Controlar roedores y aves durante madurez."
        )
    }
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
BASE_FONT_SIZE   = 12
BUTTON_FONT_SIZE = 10
STAGE_FONT_SIZE  = 16
DESC_FONT_SIZE   = BUTTON_FONT_SIZE
BUTTON_FONT_FAM  = "Arial"
STAGE_FONT_FAM   = "Helvetica"
DESC_FONT_FAM    = BUTTON_FONT_FAM
IMAGE_FONT_FAM   = "Helvetica"

INITIAL_SCREEN_WIDTH  = 320
INITIAL_SCREEN_HEIGHT = 480

root = tk.Tk()
root.title('Clasificador de Etapas de Frijol')
root.attributes('-zoomed', True)

default_font = tkfont.nametofont("TkDefaultFont")
default_font.configure(size=BASE_FONT_SIZE, family="Sans")
text_font    = tkfont.nametofont("TkTextFont")
text_font.configure(size=BASE_FONT_SIZE, family="Sans")
menu_font    = tkfont.nametofont("TkMenuFont")
menu_font.configure(size=BASE_FONT_SIZE, family="Sans")

root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

main_frame = tk.Frame(root, padx=4, pady=4)
main_frame.grid(row=0, column=0, sticky="nsew")
main_frame.grid_columnconfigure(0, weight=3)
main_frame.grid_columnconfigure(1, weight=2)
main_frame.grid_rowconfigure(0, weight=1)

left_frame = tk.Frame(main_frame, bd=1, relief=tk.SOLID)
left_frame.grid(row=0, column=0, sticky="nsew", padx=(0,4))
left_frame.grid_rowconfigure(0, weight=1)
left_frame.grid_columnconfigure(0, weight=1)

image_label = tk.Label(
    left_frame,
    text='Imagen',
    fg='gray',
    font=tkfont.Font(family=IMAGE_FONT_FAM, size=BASE_FONT_SIZE+4, weight='bold')
)
image_label.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

right_frame = tk.Frame(main_frame, bd=1, relief=tk.SOLID)
right_frame.grid(row=0, column=1, sticky="nsew", padx=(4,0))
right_frame.grid_rowconfigure(0, weight=0)
right_frame.grid_rowconfigure(1, weight=0)
right_frame.grid_rowconfigure(2, weight=1)
right_frame.grid_rowconfigure(3, weight=0)
right_frame.grid_columnconfigure(0, weight=1)

stage_label = tk.Label(
    right_frame,
    text='',
    fg='blue',
    font=tkfont.Font(family=STAGE_FONT_FAM, size=STAGE_FONT_SIZE, weight='bold'),
    anchor='nw',
    justify=tk.LEFT
)
stage_label.grid(row=0, column=0, sticky="nw", padx=8, pady=(8,4))

# Descripción
desc_label = tk.Label(
    right_frame,
    text='Esperando acción...',
    font=tkfont.Font(family=DESC_FONT_FAM, size=DESC_FONT_SIZE),
    anchor='nw',
    justify=tk.LEFT,
    wraplength=(INITIAL_SCREEN_WIDTH*2//5)-10
)
desc_label.grid(row=1, column=0, sticky="nsew", padx=8)

# Recomendaciones
reco_label = tk.Label(
    right_frame,
    text='',
    font=tkfont.Font(family=DESC_FONT_FAM, size=DESC_FONT_SIZE),
    fg='green',
    anchor='nw',
    justify=tk.LEFT,
    wraplength=(INITIAL_SCREEN_WIDTH*2//5)-10
)
reco_label.grid(row=2, column=0, sticky="nsew", padx=8, pady=(4,8))

button_frame = tk.Frame(right_frame)
button_frame.grid(row=3, column=0, sticky="sew", padx=5, pady=5)
button_frame.grid_columnconfigure(0, weight=1)

capture_btn = tk.Button(
    button_frame,
    text='Tomar Foto',
    font=tkfont.Font(family=BUTTON_FONT_FAM, size=BUTTON_FONT_SIZE, weight='bold'),
    command=lambda: tomar_y_clasificar()
)
capture_btn.grid(row=0, column=0, sticky="ew", pady=(0,4))

clear_btn = tk.Button(
    button_frame,
    text='Limpiar',
    font=tkfont.Font(family=BUTTON_FONT_FAM, size=BUTTON_FONT_SIZE),
    command=lambda: limpiar(),
    state=tk.DISABLED
)
clear_btn.grid(row=1, column=0, sticky="ew")

last_photo_path = None
current_pil_image = None
resize_job = None

def classify_image(path_or_pil_image):
    if isinstance(path_or_pil_image, str):
        img = Image.open(path_or_pil_image).convert('RGB')
    else:
        img = path_or_pil_image.convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        _, pred = torch.max(out, 1)
    return class_names[pred.item()]

def display_image(pil_img_to_display):
    global current_pil_image
    pil_img_to_display = pil_img_to_display.rotate(-90, expand=True)
    current_pil_image = pil_img_to_display
    image_label.update_idletasks()
    max_w = image_label.winfo_width() - 10
    max_h = image_label.winfo_height() - 10
    img_copy = current_pil_image.copy()
    img_copy.thumbnail((max(max_w,10), max(max_h,10)), Image.LANCZOS)
    tk_img = ImageTk.PhotoImage(img_copy)
    image_label.config(image=tk_img, text='')
    image_label.image = tk_img

def tomar_y_clasificar():
    global last_photo_path
    stage_label.config(text='')
    desc_label.config(text='Capturando imagen...')
    reco_label.config(text='')
    root.update_idletasks()

    save_dir = 'fotos_campo'
    os.makedirs(save_dir, exist_ok=True)
    ruta = os.path.join(save_dir, f'captura_{datetime.now():%Y%m%d_%H%M%S}.jpg')
    try:
        subprocess.run(
            ['libcamera-jpeg','-n','-o',ruta,'-t','200','--width','1280','--height','960'],
            check=True
        )
    except Exception as e:
        desc_label.config(text=f'Error captura: {e}')
        return

    try:
        pil_image = Image.open(ruta)
        display_image(pil_image)
    except Exception as e:
        desc_label.config(text=f'Error mostrar imagen: {e}')
        return

    desc_label.config(text='Clasificando...')
    root.update_idletasks()
    try:
        cls = classify_image(pil_image)
        datos = class_recomendaciones.get(cls, {})
        stage_label.config(text=f"Etapa: {cls}")
        desc_label.config(text=datos.get("descripcion","Sin descripción."))
        reco_label.config(text=datos.get("recomendaciones","Sin recomendaciones."))
    except Exception as e:
        desc_label.config(text=f'Error clasificar: {e}')

    capture_btn.config(state=tk.DISABLED)
    clear_btn.config(state=tk.NORMAL)
    last_photo_path = ruta

def limpiar():
    global last_photo_path, current_pil_image
    last_photo_path = None
    current_pil_image = None
    image_label.config(image=None, text='Imagen', fg='gray',
                       font=tkfont.Font(family=IMAGE_FONT_FAM, size=BASE_FONT_SIZE+4, weight='bold'))
    image_label.image = None
    stage_label.config(text='')
    desc_label.config(text='Esperando acción...')
    reco_label.config(text='')
    capture_btn.config(state=tk.NORMAL)
    clear_btn.config(state=tk.DISABLED)

def update_status_wraplength(event=None):
    new_width = right_frame.winfo_width() - 30
    if new_width > 0:
        desc_label.config(wraplength=new_width)
        reco_label.config(wraplength=new_width)

def on_image_label_configure(event):
    global resize_job
    if current_pil_image:
        if resize_job:
            root.after_cancel(resize_job)
        resize_job = root.after(250, actual_image_resize_on_configure)

def actual_image_resize_on_configure():
    if not current_pil_image or not image_label.winfo_exists():
        return
    max_w = image_label.winfo_width() - 10
    max_h = image_label.winfo_height() - 10
    img_copy = current_pil_image.copy()
    img_copy.thumbnail((max(max_w,10), max(max_h,10)), Image.LANCZOS)
    tk_img = ImageTk.PhotoImage(img_copy)
    image_label.config(image=tk_img)
    image_label.image = tk_img

right_frame.bind("<Configure>", update_status_wraplength)
image_label.bind("<Configure>", on_image_label_configure)

root.update_idletasks()
update_status_wraplength()

root.mainloop()