import os
from os.path import join
import argparse
import cv2
from tqdm import tqdm

def extract_frames(video_path, output_folder_path):
    # Membuat folder output untuk frame-frame dari video ini
    os.makedirs(output_folder_path, exist_ok=True)
    
    try:
        reader = cv2.VideoCapture(video_path)
        frame_num = 0
        while reader.isOpened():
            success, image = reader.read()
            if not success:
                break
            # Simpan frame dengan format nama 0000.png, 0001.png, dst.
            cv2.imwrite(join(output_folder_path, '{:04d}.png'.format(frame_num)), image)
            frame_num += 1
        reader.release()
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")

def process_all_videos(input_dir, output_dir):
    print(f"Mencari video di dalam: {input_dir}")
    video_files_to_process = []
    # Menggunakan os.walk untuk mencari semua file secara rekursif
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_files_to_process.append(join(root, file))
    
    print(f"Ditemukan {len(video_files_to_process)} file video untuk diproses.")

    for video_path in tqdm(video_files_to_process, desc="Mengekstrak Frame"):
        # Hapus ekstensi .mp4 untuk nama folder
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        # Dapatkan path subdirektori relatif (misal: "Celeb-real")
        relative_path = os.path.relpath(os.path.dirname(video_path), input_dir)
        
        # Gabungkan untuk membuat path output akhir
        output_folder_path = join(output_dir, relative_path, video_id)
        
        # Panggil fungsi ekstraksi
        extract_frames(video_path, output_folder_path)

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Skrip generik untuk mengekstrak frame video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--input_dir', type=str, required=True,
                   help='Direktori root yang berisi video-video mentah (bisa di dalam sub-folder).')
    p.add_argument('--output_dir', type=str, required=True,
                   help='Direktori root tempat menyimpan frame-frame yang diekstrak.')
    args = p.parse_args()

    process_all_videos(args.input_dir, args.output_dir)
    print("Ekstraksi frame selesai.")