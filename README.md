# AMT-APC

AMT-APC is a method to train an automatic piano cover generation model by fine-tuning a AMT (Automatic Music Transcription) model.

- Project page: [AMT-APC](https://misya11p.github.io/amt-apc/)
- Paper: [AMT-APC: ](https://arxiv.org/abs/)

## Usage (Piano Cover Generation)

1. Install dependencies

```bash
pip install -r requirements.txt
```

or, if you only execute the inference code, you can install only the necessary packages.

```bash
pip install torch torchaudio pretty-midi tqdm
```

2. Download the pre-trained model

```bash
wget
```

3. Run the inference code

```bash
python infer input.wav
```

You can also input the YouTube URL (requires [`yt-dlp`](https://github.com/yt-dlp/yt-dlp)).

```bash
python infer 'https://www.youtube.com/watch?v=...'
```

You can also specify the style (`level1`, `level2`, `level3`).

```bash
python infer input.wav --style level3
```

## Usage (Training & Evaluation)

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Download the dataset

```bash
python download.py
```

The dataset directory is `dataset/` by default. You can change the directory by changing `path.dataset` in `config.json`.

3. Create the dataset

```bash
python data/sync.py
python data/transcribe.py
python data/sv/extract.py
python data/create_labels.py
python data/create_dataset.py
```

4. Train the model

```bash
python train --n_gpus 1
```

5. Evaluate the model

```bash
python eval/cover.py
python eval/distance.py
```

### Options

Detailed configuration can be done via `config.json`, or using command line options, which are explained with `--help`. These default values ​​are the ones used in the experiments in the paper.
