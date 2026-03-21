## Parameter-Efficient Adaptation for MLLMs via Implicit Modality Decomposition

#### Requirements and Installation
* Python == 3.10
* Pytorch == 2.0.0
* transformers==4.37.2
* deepspeed == 0.12.6

Install required packages:
```bash
conda create -n imod python=3.10 -y
conda activate imod
pip install -r requirements.txt
```

#### Download pretrained weights
* Visual encoder: [openai-clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
* Audio encoder: [Fine-tuned BEATs_iter3+ (AS2M)](https://github.com/microsoft/unilm/blob/master/beats/README.md)
* LLM: [LLaMA-2-Chat-HF](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

Set the path of `vit_ckpt_path`, `BEATs_ckpt_path` and `model_name_or_path` in pretrain and finetune scripts.

#### Prepare datasets
* Download image and video pretrain dataset from [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA/blob/main/TRAIN_AND_VALIDATE.md)
* Download audio pretrain dataset from [AudioCaps](https://github.com/cdjkim/audiocaps)
* Download the [MUSIC-AVQA](https://github.com/GeWu-Lab/MUSIC-AVQA) dataset and extract audio waveforms from videos. 

Set the path of pretrain dataset at:
```
dataset/pretrain_dataset.py
```
Set the path of finetuning dataset at
```
dataset/unified_dataset.py
```

#### Training
Replac path of [google-bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) in `models/multimodal_encoder.py`. All experiments are conducted on 8 A100 80G GPUs.

**Stage 1: Pre-train projectors**

```bash
sh scripts/pretrain/pretrain_visual.sh
sh scripts/pretrain/pretrain_audio.sh
```
**Stage 2: Fine-tuning**

Set the path of pre-trained projectors at:

```bash
sh scripts/finetune/finetune.py
```
Fine-tuning:
```bash
sh scripts/finetune/ft.sh
```

#### Inference

Set the `CKPT_DIR` path and inference:

```bash
sh scripts/finetune/infer.sh
```

#### Evaluation

Set the `result_path` and evaluate:

```bash
python evaluation.py
```

#### Acknowledgement
This codebase is largely based on the [MokA](https://github.com/GeWu-Lab/MokA/tree/main). We also would like to thank the following repositories:

* [MokA](https://github.com/GeWu-Lab/MokA/tree/main)

* [Crab](https://github.com/GeWu-Lab/Crab?tab=readme-ov-file)
