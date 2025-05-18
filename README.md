## Installation

uv sync
source ./venv/bin/activate

## Run
### Training
python pet_train.py --mode baseline --epoch 30 --batch 32
python pet_train.py --mode masked --epoch 30 --batch 32
python pet_train.py --mode full_aug --epoch 30 --batch 32
python pet_train.py --mode background_aug --epoch 30 --batch 32

### Evaluation
python pet_eval.py --ckpts runs/..._baseline/e30.pth runs/..._masked/e30.pth runs/..._full_aug/e30.pth runs/..._background_aug/e30.pth --out eval_log_30.csv

### Visualiztions
python visual.py \
    --data ./data \
    --runs runs/..._baseline runs/..._masked runs/..._full_aug runs/..._background_aug \
    --sample_img ./data/oxford-iiit-pet/images/american_pit_bull_terrier_55.jpg \
    --eval_csv ./eval_log_30.csv \
    --size 384 \
    --out visualizations/30
