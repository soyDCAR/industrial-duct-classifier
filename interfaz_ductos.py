import json
import os
import tkinter as tk
from tkinter import filedialog

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageTk
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


# -------------------- Modelo --------------------
class MultiEfficientNet(torch.nn.Module):
    def __init__(self, base_model, num_classes_d, num_classes_o):
        super().__init__()
        self.features = torch.nn.Sequential(*list(base_model.children())[:-1])
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = torch.nn.Flatten()
        self.classifier_d = torch.nn.Linear(base_model.classifier[1].in_features, num_classes_d)
        self.classifier_o = torch.nn.Linear(base_model.classifier[1].in_features, num_classes_o)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        out_d = self.classifier_d(x)
        out_o = self.classifier_o(x)
        return out_d, out_o

# -------------------- Cargar modelo y clases --------------------
device = torch.device("cpu")  # Forzar CPU

# Cargar mapeo desde JSON
with open("class_mapping.json") as f:
    mapping = json.load(f)
    idx_to_class_d = {int(k): v for k, v in mapping["d"].items()}
    idx_to_class_o = {int(k): v for k, v in mapping["o"].items()}

# Cargar modelo
base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model = MultiEfficientNet(base, len(idx_to_class_d), len(idx_to_class_o))
model.load_state_dict(torch.load("modelo_ductos_multitarea_efnet.pth", map_location=device))
model.to(device)
model.eval()

# Transformación de entrada
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------- Interfaz Tkinter --------------------
class App:
    def __init__(self, master):
        self.master = master
        master.title("🔍 Clasificador de Imágenes de Ductos")
        self.frame = tk.Frame(master)
        self.frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Botón de carga
        self.btn_cargar = tk.Button(self.frame, text="📂 Cargar Imágenes", command=self.cargar_imagenes, font=("Arial", 12))
        self.btn_cargar.pack(pady=10)

        # Canvas con scrollbar
        self.canvas = tk.Canvas(self.frame, width=900, height=600)
        self.scroll_y = tk.Scrollbar(self.frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scroll_y.set)
        self.scroll_y.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.img_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.img_frame, anchor="nw")

        self.img_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

    def cargar_imagenes(self):
        rutas = filedialog.askopenfilenames(filetypes=[("Imágenes", "*.jpg *.jpeg *.png")])
        if not rutas:
            return

        for widget in self.img_frame.winfo_children():
            widget.destroy()

        for ruta in rutas:
            self.mostrar_y_predecir(ruta)

    def mostrar_y_predecir(self, ruta_img):
        try:
            img = Image.open(ruta_img).convert("RGB")
        except Exception:
            return

        # Mostrar miniatura
        img_mini = img.copy()
        img_mini.thumbnail((200, 200))
        tk_img = ImageTk.PhotoImage(img_mini)

        lbl_img = tk.Label(self.img_frame, image=tk_img)
        lbl_img.image = tk_img
        lbl_img.pack(pady=5)

        # Predicción
        img_tensor = transform_val(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out_d, out_o = model(img_tensor)
            pred_d = torch.argmax(out_d, dim=1).item()
            pred_o = torch.argmax(out_o, dim=1).item()

        # Mapear a clase real
        d_real = idx_to_class_d[pred_d]
        o_real = idx_to_class_o[pred_o]
        v_real = max(d_real - o_real, 0)

        # Mostrar resultados
        lbl_result = tk.Label(
            self.img_frame,
            text=f"{os.path.basename(ruta_img)}\n"
                 f"🔢 Ductos totales (dX): {d_real if d_real < 7 else '7+'}\n"
                 f"✅ Ductos ocupados (oX): {o_real if o_real < 7 else '7+'}\n"
                 f"⬜ Ductos vacíos     : {v_real}",
            font=("Arial", 10)
        )
        lbl_result.pack(pady=(0, 15))

# -------------------- Ejecutar Interfaz --------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
