# DLA HW3 TTS project

## Installation guide
Отдельные скрипты для тренировочного и тестового окружения:
- `scripts/train_install.sh` (здесь есть загрузка LJSpeech и предподсчитанных характеристик)
- `scripts/infer_install.sh` (здесь есть загрузка WaveGlow)

## Preprocessing
Везде используются параметры и пути из `configs.py`.

- Подсчет energy и pitch `python preprocess.py calculate_energy_pitch`
  Скорее всего займёт 35-45 минут на одном ядре, но внутри есть распараллеливание на потоки через joblib, поэтому рекомендуется запускать на машинке с бОльшим количеством ядер.
- Подсчет максимумов и минимумов `python preprocess.py calculate_min_max`
- Нормализация `python preprocess.py normalize`

## Inference guide
Для синтеза можно использовать `synthesize.py`, передавая путь к модели, тексты и variance-характеристики. По тексту и variance будет построено декартово произведение, для каждого его элемента (набора парамтеров) будет проведён синтез. Пример запуска можно увидеть в `scripts/synthesize.sh`.
```bash
python synthesize.py \
    --checkpoint-path="./model_new/current_model-state_dict-N.pth" \
    --text \
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest" \
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education" \
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space" \
    --duration-alpha 0.8 1.0 1.2 \
    --energy-alpha 0.8 1.0 1.2 \
    --pitch-alpha 0.8 1.0 1.2
```

## Examples of synthesis
Папка results, где используются тексты:
1. A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest
2. Massachusetts Institute of Technology may be best known for its math, science and engineering education
3. Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space

## Report
[wandb](https://wandb.ai/danwallgun/fastspeech_example/reports/DLA-HW2-TTS-Report--VmlldzozMDQ5NTY5)

## Credits
- https://github.com/markovka17/dla/tree/2022/hw3_tts
- https://github.com/markovka17/dla/blob/2022/week07/FastSpeech_sem.ipynb
- https://github.com/xcmyz/FastSpeech/
- https://arxiv.org/abs/1905.09263
- https://arxiv.org/abs/2006.04558v3