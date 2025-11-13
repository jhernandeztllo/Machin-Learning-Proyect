# ===============================================
# RECICLAJE INTELIGENTE - MAIN (adaptado a tu proyecto)
# ===============================================

from tkinter import *
from PIL import Image, ImageTk
import imutils
import cv2
import numpy as np
from ultralytics import YOLO
import os
import math

# ---- Limpiar labels ----
def clean_lbl():
    lblimg.config(image='')
    lblimgtxt.config(image='')

# ---- Mostrar imágenes de clase ----
def images(img, imgtxt):
    img = np.array(img, dtype="uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = Image.fromarray(img)
    img_ = ImageTk.PhotoImage(image=img)
    lblimg.configure(image=img_)
    lblimg.image = img_

    imgtxt = np.array(imgtxt, dtype="uint8")
    imgtxt = cv2.cvtColor(imgtxt, cv2.COLOR_BGR2RGB)
    imgtxt = Image.fromarray(imgtxt)
    img_txt = ImageTk.PhotoImage(image=imgtxt)
    lblimgtxt.configure(image=img_txt)
    lblimgtxt.image = img_txt

# ---- Escaneo en tiempo real ----
def Scanning():
    global lblimg, lblimgtxt
    detect = False

    if cap is not None:
        ret, frame = cap.read()
        frame_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if ret:
            results = model(frame, stream=True, verbose=False)
            for res in results:
                boxes = res.boxes
                for box in boxes:
                    detect = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Corrige límites negativos
                    x1, y1 = max(x1, 0), max(y1, 0)

                    # Dibujar cuadro
                    color = class_colors[cls]
                    label = f'{clsName[cls]} {conf:.2f}'
                    cv2.rectangle(frame_show, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_show, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    # Mostrar imagen correspondiente
                    if cls == 0:
                        images(img_glass, img_glasstxt)
                    elif cls == 1:
                        images(img_metal, img_metaltxt)
                    elif cls == 2:
                        images(img_paper, img_papertxt)
                    elif cls == 3:
                        images(img_plastic, img_plastictxt)

            if not detect:
                clean_lbl()

            frame_show = imutils.resize(frame_show, width=640)
            im = Image.fromarray(frame_show)
            img = ImageTk.PhotoImage(image=im)
            lblVideo.configure(image=img)
            lblVideo.image = img
            lblVideo.after(10, Scanning)
        else:
            cap.release()

# ---- Interfaz principal ----
def ventana_principal():
    global cap, lblVideo, model, clsName, class_colors
    global img_glass, img_metal, img_paper, img_plastic
    global img_glasstxt, img_metaltxt, img_papertxt, img_plastictxt
    global lblimg, lblimgtxt, pantalla

    pantalla = Tk()
    pantalla.title("RECICLAJE INTELIGENTE")
    pantalla.geometry("1280x720")

    # Fondo
    imagenF = PhotoImage(file="setUp/Canva.png")
    background = Label(image=imagenF)
    background.place(x=0, y=0, relwidth=1, relheight=1)

    # Modelo entrenado (ajusta ruta si es necesario)
    model = YOLO(r"D:\Jotaaaa Documentos\UNI\Maching Learingn\Reciclaje2\reciclaje_ai-main\runs\detect\train5\weights\best.pt")
    # tu modelo entrenado con glass, metal, paper, plastic

    # Clases según tu dataset
    clsName = ['glass', 'metal', 'paper', 'plastic']

    # Colores por clase (RGB)
    class_colors = [
        (0, 255, 255),  # glass
        (255, 255, 0),  # metal
        (0, 255, 0),    # paper
        (0, 0, 255)     # plastic
    ]

    # Imágenes ilustrativas (ajusta según tus nombres reales)
    # 📁 Detectar rutas
    base_dir = os.path.dirname(os.path.abspath(__file__))
    setup_dir = os.path.join(base_dir, "setUp")

    # Construir rutas absolutas
    path_glass      = os.path.join(setup_dir, "glass.png")
    path_metal      = os.path.join(setup_dir, "metal.png")
    path_paper      = os.path.join(setup_dir, "paper.png")
    path_plastic    = os.path.join(setup_dir, "plastic.png")
    path_glasstxt   = os.path.join(setup_dir, "glasstxt.png")
    path_metaltxt   = os.path.join(setup_dir, "metaltxt.png")
    path_papertxt   = os.path.join(setup_dir, "papertxt.png")
    path_plastictxt = os.path.join(setup_dir, "plastictxt.png")

    # Mostrar rutas para verificar
    print("[INFO] Verificando rutas:")
    for p in [path_glass, path_metal, path_paper, path_plastic,
            path_glasstxt, path_metaltxt, path_papertxt, path_plastictxt]:
        print("  ", p, "✅" if os.path.exists(p) else "❌")

    # Cargar imágenes
    img_glass      = cv2.imread(path_glass)
    img_metal      = cv2.imread(path_metal)
    img_paper      = cv2.imread(path_paper)
    img_plastic    = cv2.imread(path_plastic)
    img_glasstxt   = cv2.imread(path_glasstxt)
    img_metaltxt   = cv2.imread(path_metaltxt)
    img_papertxt   = cv2.imread(path_papertxt)
    img_plastictxt = cv2.imread(path_plastictxt)

    # Video
    lblVideo = Label(pantalla)
    lblVideo.place(x=320, y=180)

    lblimg = Label(pantalla)
    lblimg.place(x=75, y=260)

    lblimgtxt = Label(pantalla)
    lblimgtxt.place(x=995, y=310)

    # Cámara
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 1280)
    cap.set(4, 720)

    Scanning()
    pantalla.mainloop()


if __name__ == "__main__":
    ventana_principal()
