# AMT-APC

AMT-APC is a method for training an automatic piano cover generation model by fine-tuning an AMT (Automatic Music Transcription) model.

- Project page: [AMT-APC](https://misya11p.github.io/amt-apc/)
- Paper: [[2409.14086] AMT-APC: Automatic Piano Cover by Fine-Tuning an Automatic Music Transcription Model](https://arxiv.org/abs/2409.14086)

## Usage (Piano Cover Generation)

Python version: 3.10

1. Install dependencies

```bash
pip install -r requirements.txt
```

Alternatively, if you only need to run the inference code, you can install just the necessary packages.

```bash
pip install torch torchaudio pretty-midi tqdm
```

2. Download the pre-trained model

```bash
wget -P models/params/ https://github.com/misya11p/amt-apc/releases/download/beta/apc.pth
```

3. Run the inference code

```bash
python infer input.wav
```

You can also input a YouTube URL (requires [`yt-dlp`](https://github.com/yt-dlp/yt-dlp)).

```bash
python infer 'https://www.youtube.com/watch?v=...'
```

You can also specify a style (`level1`, `level2`, `level3`).

```bash
python infer input.wav --style level3
```

## Usage (Training & Evaluation)

Python version: 3.10

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Download the pre-trained AMT model

```bash
wget -P models/params/ https://github.com/misya11p/amt-apc/releases/download/beta/amt.pth
```

3. Download the dataset

```bash
python download.py
```

The dataset directory is set to `dataset/` by default. You can change this directory by modifying `path.dataset` in `config.json`.

4. Create the dataset

```bash
python data/sync.py
python data/transcribe.py
python data/sv/extract.py
python data/create_labels.py
python data/create_dataset.py
```

5. Train the model

```bash
python train --n_gpus 1
```

5. Evaluate the model

Calculate $\mathcal Q_{\text{max}}$.

```bash
git clone https://github.com/albincorreya/ChromaCoverId.git eval/ChromaCoverId
python eval/cover.py
python eval/distance.py
```

### Options

Detailed configuration can be done through `config.json` or by using command line options, which are explained with --help. The default values are those used in the experiments in the paper.
