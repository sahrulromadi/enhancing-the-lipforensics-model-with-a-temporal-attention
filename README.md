# IMPROVING THE LIPFORENSICS DEEPFAKE DETECTION MODEL USING A TEMPORAL ATTENTION MECHANISM

Temporal Attention is a smart mechanism 🧠 in artificial intelligence models, especially for analyzing video or other sequential data. This mechanism allows the model to focus on the most important moments in a time sequence, rather than treating every video frame with the same weight.

In simple terms, here's how it works:

1. The model "watches" the entire video clip.
2. It then gives an "importance score" to each frame. A frame where a strange lip movement occurs might get a high score, while a still frame gets a low score.
3. These scores are then used to create a video summary that is biased towards the important frames.

As a result, the model can make more accurate decisions because it learns to pay attention to the most relevant parts and ignore the uninformative ones.

## Setup

### Install packages

```bash
pip install -r requirements.txt
```

Note: we used Python version 3.8 to test this code.

### Prepare data

1. Follow the links below to download the datasets:

   - [CelebDF](https://github.com/yuezunli/celeb-deepfakeforensics)

2. Extract the frames with run

```bash
python preprocessing/extract_video.py
```

The filenames of the frames should be as follows: 0000.png, 0001.png, ....

3. Detect the faces and compute 68 face landmarks, you can run

```bash
python preprocessing/detect_landmarks.py
```

4. Place face frames and corresponding landmarks into the appropriate directories:

   - Put the frames here: data/datasets/CelebDF/{dataset_name}/images/{video}. Use RealCelebDF for real videos and FakeCelebDF for fake videos from the test set.

5. To crop the mouth region from each frame for all datasets, run
   ```bash
   python preprocessing/crop_mouths.py --dataset all
   ```
   This will write the mouth images into the corresponding `cropped_mouths` directory.

## Finetuning

1. Download the [pretrained model](https://drive.google.com/file/d/1wfZnxZpyNd5ouJs0LjVls7zU0N_W73L7/view?usp=sharing) and place into `models/weights`. This model has been trained on FaceForensics++ (Deepfakes, FaceSwap, Face2Face, and NeuralTextures)

2. Finetune this model on another dataset, for example, CelebDF-v2. The command below will load the existing weights, continue training, and then save it as a new model.

```bash
python train.py --dataset CelebDF --weights_forgery ./models/weights/lipforensics_ff.pth --save_path ./models/weights/lipforensics_finetuned.pth
```

## Evaluate

After the finetuning process is complete, evaluate your new model to see its improvement. Use the evaluate.py script, pointing it to the weights from the finetuning results:

```bash
python evaluate.py --dataset CelebDF --weights_forgery ./models/weights/lipforensics_finetuned.pth
```

Here is a sample comparison of the model's performance on the CelebDF-v2 dataset, before and after applying the temporal attention mechanism and finetuning process.

| Metric | Before (Original Model) | After (Temporal Attention) |
| ------ | ----------------------- | -------------------------- |
| AUC    | 0.78                    | 0.85                       |

## Citation

If you find this repo useful for your research, please consider citing the following:

```bibtex
@inproceedings{haliassos2021lips,
  title={Lips Don't Lie: A Generalisable and Robust Approach To Face Forgery Detection},
  author={Haliassos, Alexandros and Vougioukas, Konstantinos and Petridis, Stavros and Pantic, Maja},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5039--5049},
  year={2021}
}
```
