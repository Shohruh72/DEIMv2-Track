## Person detector: fine-tune on CrowdHuman + MOT17

A 1-class `person` model for tracking, fine-tuned from the COCO `s` checkpoint.

* train: CrowdHuman train (15,000 images) + MOT17 train, first half of every sequence (2,657 frames)
* val: CrowdHuman val (4,370 images); the second halves of MOT17 stay unseen for the tracking evaluation

### 1. Folder structure

```
tracking/
├── DEIMv2-pt/
│   └── weights/safetensors/deimv2_s.safetensors
└── Dataset/
    ├── CrowdHuman/
    │   ├── Images/                  # *.jpg
    │   ├── annotation_train.odgt
    │   └── annotation_val.odgt
    ├── MOT17/
    │   ├── det/train/MOT17-XX/img1/  # *.jpg
    │   └── labels/train/MOT17-XX-FRCNN/{gt/gt.txt, seqinfo.ini}
    └── Person/
        ├── convert.py
        └── data.yaml                # classes and the fine-tune schedule (24 epochs)
```

Copying to another machine: copy only `convert.py` and `data.yaml` of `Person/`,
its `images/` are symlinks with absolute paths.

### 2. Convert (once)

```bash
cd Dataset/Person
python convert.py   # YOLO-format labels + symlinked images -> train.txt, val.txt
```

Re-converting: delete `images/`, `labels/` and `images/*.cache` first.

### 3. Train

```bash
cd DEIMv2-pt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python main.py --model s --train --data ../Dataset/Person/data.yaml \
               --weights weights/safetensors/deimv2_s.safetensors --batch-size 4
```

* multi-GPU: `bash main.sh 2 --model s --train ...` (same arguments)
* `--batch-size` is per GPU, gradients are accumulated to a total batch of 32:
  4 for 8 GB, 8 for 16 GB, 16 for 24 GB+
* add `--resume` to continue an interrupted run from `weights/pt/stage.pt`
* outputs: `weights/pt/best.pt`, `weights/pt/last.pt`, `weights/pt/step.csv` (loss and mAP per epoch)
* time: ~13 min / epoch with batch 4 on an 8 GB GPU (~5 h in total)

| Epoch | 1     | 5     | 9     | 11    |
|:-----:|:-----:|:-----:|:-----:|:-----:|
| mAP   | 0.440 | 0.505 | 0.522 | 0.526 |

### 4. Use

Always pass `--data`, it sets the class count (1) and names:

```bash
python main.py --model s --test --data ../Dataset/Person/data.yaml --weights weights/pt/best.pt
python main.py --model s --demo --track --data ../Dataset/Person/data.yaml --weights weights/pt/best.pt
python main.py --model s --export engine --data ../Dataset/Person/data.yaml --weights weights/pt/best.pt
```

### License

CrowdHuman ([CC BY-NC 4.0](https://www.crowdhuman.org)) and MOT17 ([CC BY-NC-SA 3.0](https://motchallenge.net))
are non-commercial, so is a model trained on them.
