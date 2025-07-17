import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import face_alignment 
import cv2

def main(args):
    # --- Inisialisasi Model ---
    # Pilih device (GPU jika tersedia, jika tidak CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Menggunakan device: {device}")

    # Muat model face_alignment. Model ini akan menangani deteksi wajah dan landmark.
    # LandmarksType.TWO_D akan menghasilkan 68 titik landmark.
    print("Memuat model Face Alignment (FAN)...")
    try:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)
    except Exception as e:
        print(f"Gagal memuat model face-alignment. Pastikan sudah terinstal dengan benar. Error: {e}")
        return
    print("Model FAN berhasil dimuat.")

    # --- Mulai Pemrosesan ---
    print(f"Mulai memproses direktori input: {args.input_dir}")
    
    # Cari semua file gambar (.png, .jpg) secara rekursif
    image_paths = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        print(f"Peringatan: Tidak ada file gambar yang ditemukan di {args.input_dir}")
        return

    # Proses setiap gambar dengan progress bar
    for image_path in tqdm(image_paths, desc="Mendeteksi Landmark"):
        try:
            # Baca gambar menggunakan OpenCV
            img = cv2.imread(image_path)
            if img is None:
                print(f"Peringatan: Gagal membaca gambar {image_path}, dilewati.")
                continue

            # --- Deteksi Landmark ---
            # Fungsi get_landmarks akan secara otomatis mendeteksi wajah terlebih dahulu.
            # Ia mengembalikan sebuah LIST dari array landmark (satu array untuk setiap wajah yang terdeteksi).
            landmarks_list = fa.get_landmarks(img)
            
            # Jika tidak ada wajah yang terdeteksi, lewati frame ini
            if landmarks_list is None:
                print(f"Peringatan: Tidak ada wajah yang terdeteksi di {image_path}, dilewati.")
                continue

            # Ambil landmark dari wajah pertama yang terdeteksi
            landmarks = landmarks_list[0] # Hasilnya adalah array numpy (68, 2)

            # --- Simpan Hasil Landmark ---
            # Buat path output yang sesuai dengan struktur input
            relative_path = os.path.relpath(image_path, args.input_dir)
            # Ganti ekstensi file gambar (.png) menjadi .npy
            output_path_without_ext = os.path.join(args.output_dir, os.path.splitext(relative_path)[0])
            
            # Buat direktori output jika belum ada
            os.makedirs(os.path.dirname(output_path_without_ext), exist_ok=True)
            
            # Simpan array landmark sebagai file .npy
            np.save(output_path_without_ext + '.npy', landmarks)

        except Exception as e:
            print(f"Terjadi error saat memproses {image_path}: {e}")

    print("Deteksi landmark selesai.")

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Skrip Tahap 2: Deteksi wajah dan landmark menggunakan face-alignment.")
    p.add_argument('--input_dir', '-i', type=str, required=True, help='Direktori berisi frame-frame gambar hasil Tahap 1.')
    p.add_argument('--output_dir', '-o', type=str, required=True, help='Direktori untuk menyimpan file-file landmark .npy.')
    args = p.parse_args()
    
    main(args)