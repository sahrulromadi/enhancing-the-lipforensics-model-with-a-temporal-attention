import argparse
from collections import defaultdict

import pandas as pd
from sklearn import metrics
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop
from tqdm import tqdm

from data.transforms import NormalizeVideo, ToTensorVideo
from data.dataset_clips import ForensicsClips, CelebDFClips, DFDCClips
from data.samplers import ConsecutiveClipSampler
from models.spatiotemporal_net import get_model
from utils import get_files_from_split


def parse_args():
    parser = argparse.ArgumentParser(description="DeepFake detector evaluation")
    parser.add_argument(
        "--dataset",
        help="Dataset to evaluate on",
        type=str,
        choices=[
            "FaceForensics++",
            "Deepfakes",
            "FaceSwap",
            "Face2Face",
            "NeuralTextures",
            "FaceShifter",
            "DeeperForensics",
            "CelebDF",
            "DFDC",
        ],
        default="cuda:0",
    )
    parser.add_argument(
        "--compression",
        help="Video compression level for FaceForensics++",
        type=str,
        choices=["c0", "c23", "c40"],
        default="c23",
    )
    parser.add_argument("--grayscale", dest="grayscale", action="store_true")
    parser.add_argument("--rgb", dest="grayscale", action="store_false")
    parser.set_defaults(grayscale=True)
    parser.add_argument("--frames_per_clip", default=25, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--device", help="Device to put tensors on", type=str, default="cuda:0")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--weights_forgery_path",
        help="Path to pretrained weights for forgery detection",
        type=str,
        default="./models/weights/lipforensics_ff.pth"
    )
    parser.add_argument(
        "--split_path", help="Path to FF++ splits", type=str, default="./data/datasets/Forensics/splits/test.json"
    )
    parser.add_argument(
        "--dfdc_metadata_path", help="Path to DFDC metadata", type=str, default="./data/datasets/DFDC/metadata.json"
    )

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    model = get_model(weights_forgery_path=args.weights_forgery_path)

    # Definisikan Optimizer Adam
    # Kita hanya akan melatih parameter yang tidak dibekukan (lapisan atensi dan klasifikasi)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.0001)

    # Definisikan Loss Function untuk klasifikasi biner
    criterion = torch.nn.BCEWithLogitsLoss()

    # Get dataset
    transform = Compose(
        [ToTensorVideo(), CenterCrop((88, 88)), NormalizeVideo((0.421,), (0.165,))]
    )
    if args.dataset in [
        "FaceForensics++",
        "Deepfakes",
        "FaceSwap",
        "Face2Face",
        "NeuralTextures",
        "FaceShifter",
        "DeeperForensics",
    ]:
        if args.dataset == "FaceForensics++":
            fake_types = ("Deepfakes", "FaceSwap", "Face2Face", "NeuralTextures")
        else:
            fake_types = (args.dataset,)

        test_split = pd.read_json(args.split_path, dtype=False)
        test_files_real, test_files_fake = get_files_from_split(test_split)

        dataset = ForensicsClips(
            test_files_real,
            test_files_fake,
            args.frames_per_clip,
            grayscale=args.grayscale,
            compression=args.compression,
            fakes=fake_types,
            transform=transform,
            max_frames_per_video=110,
        )
    elif args.dataset == "CelebDF":
        dataset = CelebDFClips(args.frames_per_clip, args.grayscale, transform)
    else:
        metadata = pd.read_json(args.dfdc_metadata_path).T
        dataset = DFDCClips(args.frames_per_clip, metadata, args.grayscale, transform)

    # Get sampler that splits video into non-overlapping clips
    sampler = ConsecutiveClipSampler(dataset.clips_per_video)

    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)

    num_epochs = 10  # Tentukan jumlah epoch (misalnya 5-10 untuk fine-tuning)
    print(f"Memulai fine-tuning untuk {num_epochs} epoch...")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Atur model ke mode training
        model.train()

        running_loss = 0.0

        # Loop melalui setiap batch data dari data loader
        for data in tqdm(loader, desc=f"Training Epoch {epoch + 1}"):
            images, labels, _ = data  # Kita tidak butuh video_indices untuk training

            # Pindahkan data ke device yang sesuai (CPU dalam kasus Anda)
            images = images.to(args.device)
            labels = labels.to(args.device)

            # 1. Nol-kan gradien dari iterasi sebelumnya
            optimizer.zero_grad()

            # 2. Forward pass: dapatkan output dari model
            logits = model(images, lengths=[args.frames_per_clip] * images.shape[0])

            # 3. Hitung loss
            loss = criterion(logits.squeeze(1), labels.float())

            # 4. Backward pass: hitung gradien
            loss.backward()

            # 5. Update bobot model
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(loader)
        print(f"Epoch {epoch + 1} selesai, Loss: {epoch_loss:.4f}")

    # Setelah loop selesai, simpan model yang sudah di-fine-tune
    print("Pelatihan selesai. Menyimpan model...")
    save_path = "weights/finetuned_attention_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model baru disimpan di: {save_path}")


if __name__ == "__main__":
    main()
