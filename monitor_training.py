#!/usr/bin/env python3
"""
Script pour surveiller le progrès de l'entraînement
"""

import os
import time
import glob
from pathlib import Path

def monitor_training():
    """Surveille le progrès de l'entraînement"""
    training_dir = "ava_phone_training/yolo11n_ava_phone"
    
    print("=== SURVEILLANCE DE L'ENTRAÎNEMENT ===")
    print(f"Dossier d'entraînement: {training_dir}")
    
    if not os.path.exists(training_dir):
        print("Le dossier d'entraînement n'existe pas encore.")
        return
    
    # Vérifier les fichiers créés
    files_to_check = [
        "weights/best.pt",
        "weights/last.pt", 
        "results.csv",
        "results.png",
        "confusion_matrix.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "BoxPR_curve.png",
        "BoxF1_curve.png"
    ]
    
    print("\nFichiers d'entraînement:")
    for file_path in files_to_check:
        full_path = f"{training_dir}/{file_path}"
        if os.path.exists(full_path):
            size = os.path.getsize(full_path)
            print(f"  ✅ {file_path} ({size:,} bytes)")
        else:
            print(f"  ⏳ {file_path} (en cours...)")
    
    # Vérifier les logs
    log_files = glob.glob(f"{training_dir}/*.txt")
    if log_files:
        print(f"\nFichiers de log trouvés: {len(log_files)}")
        for log_file in log_files:
            print(f"  📄 {os.path.basename(log_file)}")
    
    # Vérifier les images de batch
    batch_images = glob.glob(f"{training_dir}/train_batch*.jpg") + glob.glob(f"{training_dir}/val_batch*.jpg")
    if batch_images:
        print(f"\nImages de batch: {len(batch_images)}")
        for img in batch_images:
            print(f"  🖼️ {os.path.basename(img)}")
    
    # Vérifier le fichier args.yaml
    args_file = f"{training_dir}/args.yaml"
    if os.path.exists(args_file):
        print(f"\nConfiguration d'entraînement:")
        with open(args_file, 'r') as f:
            content = f.read()
            lines = content.split('\n')[:10]  # Premières 10 lignes
            for line in lines:
                if line.strip():
                    print(f"  {line}")

def check_training_status():
    """Vérifie le statut de l'entraînement"""
    print("\n=== STATUT DE L'ENTRAÎNEMENT ===")
    
    # Vérifier si le processus Python est en cours
    import subprocess
    try:
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python3.11.exe'], 
                              capture_output=True, text=True)
        if 'python3.11.exe' in result.stdout:
            print("✅ Processus d'entraînement en cours")
        else:
            print("❌ Aucun processus d'entraînement détecté")
    except:
        print("⚠️ Impossible de vérifier les processus")
    
    # Vérifier la taille du dossier d'entraînement
    training_dir = "ava_phone_training"
    if os.path.exists(training_dir):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(training_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        
        print(f"📁 Taille du dossier d'entraînement: {total_size / 1024 / 1024:.1f} MB")
    
    # Vérifier les derniers fichiers créés
    all_files = []
    for root, dirs, files in os.walk("ava_phone_training"):
        for file in files:
            filepath = os.path.join(root, file)
            mtime = os.path.getmtime(filepath)
            all_files.append((filepath, mtime))
    
    if all_files:
        all_files.sort(key=lambda x: x[1], reverse=True)
        print(f"\n📅 Derniers fichiers créés:")
        for filepath, mtime in all_files[:5]:
            filename = os.path.basename(filepath)
            time_str = time.strftime("%H:%M:%S", time.localtime(mtime))
            print(f"  {time_str} - {filename}")

def main():
    """Fonction principale"""
    monitor_training()
    check_training_status()
    
    print(f"\n=== INSTRUCTIONS ===")
    print("• L'entraînement peut prendre plusieurs heures")
    print("• Surveillez les fichiers dans ava_phone_training/yolo11n_ava_phone/")
    print("• Le modèle sera sauvegardé dans weights/best.pt")
    print("• Relancez ce script pour vérifier le progrès")

if __name__ == "__main__":
    main()
