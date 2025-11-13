from pathlib import Path

# Ruta base del dataset (ajústala solo si cambia)
base_path = Path(r"D:\Jotaaaa Documentos\UNI\Maching Learingn\reciclaje_ai-main\reciclaje_ai-main\dataset")

# Clases YOLO con su ID
classes = {"glass": 0, "metal": 1, "paper": 2, "plastic": 3}

# Subcarpetas (train y val)
splits = ["train", "val"]

total_labels = 0

for split in splits:
    for cls_name, cls_id in classes.items():
        # Directorios de imágenes y labels
        img_dir = base_path / "images" / split / cls_name
        lbl_dir = base_path / "labels" / split / cls_name

        # Crear carpeta de labels si no existe
        lbl_dir.mkdir(parents=True, exist_ok=True)

        # Verificar existencia de carpeta de imágenes
        if not img_dir.exists():
            print(f"⚠️ No existe la carpeta de imágenes: {img_dir}")
            continue

        count = 0

        # Recorremos las imágenes
        for img_file in img_dir.glob("*.*"):
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            # Nombre base (ej: glass_0001)
            label_name = f"{img_file.stem}.txt"
            label_path = lbl_dir / label_name

            # Crear archivo YOLO
            with open(label_path, "w") as f:
                f.write(f"{cls_id} 0.5 0.5 1.0 1.0\n")

            count += 1

        total_labels += count
        print(f"✅ {count} etiquetas creadas para '{cls_name}' en '{split}'")

print(f"\n🎯 Proceso completado correctamente. Total de etiquetas creadas: {total_labels}")
