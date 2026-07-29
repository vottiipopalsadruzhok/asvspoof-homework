# ASVspoof2019 LA --- LCNN

Ниже --- воспроизведение обучения и формирования предсказаний в Kaggle-ноутбуке.

### Подключение данных

Kaggle-датасет [ASVspoof 2019 Dataset](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset), нужна папка `LA/LA`.

### Установка зависимостей

```bash
pip install -q hydra-core torchmetrics wandb
```

> [!NOTE]
> `requirements.txt` нужен для локального воспроизведения: в Kaggle torch, torchaudio, panda, numpy уже стоят.

### Скачивание репозитория

```bash
git clone https://github.com/vottiipopalsadruzhok/asvspoof-homework.git
cd asvspoof-homework
```

### (Опционально) Отключить необходимость заходить в аккаунт Wandb

```bash
export WANDB_MODE=offline
```

> [!NOTE]
> По умолчанию логи пишутся только локально, даже при заходе в аккаунт.

### Обучение

```bash
python3 train.py data_dir=/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA
```

### Формирование предсказаний

```bash
python3 inference.py \
    data_dir=/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA \
    inferencer.from_pretrained=$PWD/saved/lcnn/checkpoint-epoch10.pth
```

Печатает `eval_EER` и пишет скоры в `data/saved/asvspoof/eval.csv`.

### Воспроизводимость

- `seed: 1` в `src/configs/baseline.yaml` и `src/configs/inference.yaml`;
- менять нужно только `data_dir`, остальное переопределяется аргументами Hydra;
- обучение и инференс с одним seed на одном GPU дают один и тот же EER.
