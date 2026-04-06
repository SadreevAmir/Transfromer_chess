# Transformer Chess

Новый пайплайн проекта:

1. Позиция кодируется в тензор `(18, 8, 8)`.
2. `12` каналов описывают типы фигур по цветам.
3. Дополнительные `6` каналов кодируют `side-to-move`, castling rights и `en passant`.
4. Модель это patchless ViT по 64 клеткам доски плюс `CLS`-токен.
5. На выходе модель выдает одно число: оценку позиции для стороны, которая ходит в этой позиции.
6. Оценки из Lichess обрезаются в диапазон `[-10, 10]` пешек.
7. На инференсе перебираются все легальные ходы, каждая resulting position оценивается моделью, и выбирается ход с лучшей оценкой для текущего игрока.

## Структура

- `src/transformer_chess/encoding.py` - кодирование доски в `(18, 8, 8)`
- `src/transformer_chess/model.py` - ViT-регрессор с scalar head
- `src/transformer_chess/lichess_eval.py` - потоковое чтение Lichess `evaluations`
- `src/transformer_chess/dataset.py` - сбор шардированного value-датасета
- `src/transformer_chess/train.py` - обучение value-модели
- `src/transformer_chess/move_selection.py` - выбор хода по value afterstate
- `src/transformer_chess/inference.py` - загрузка чекпойнта и инференс

## Установка

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Источник данных

Проект рассчитан на официальный дамп Lichess `evaluations`:

- URL: `https://database.lichess.org/lichess_db_eval.jsonl.zst`
- формат: JSONL, одна позиция на строку
- для обучения берется оценка лучшего PV

## Скачать подмножество Lichess

```bash
source .venv/bin/activate
transformer-chess download-lichess-evals \
  --output data/raw/lichess_eval_1m.jsonl \
  --max-positions 1000000
```

## Собрать value-датасет

```bash
source .venv/bin/activate
transformer-chess build-dataset \
  --input data/raw/lichess_eval_1m.jsonl \
  --output-dir data/processed/value_lichess_1m \
  --max-samples 1000000 \
  --top-k 4 \
  --min-pvs 2 \
  --clip-pawns 10 \
  --shard-size 50000
```

После этого в `data/processed/value_lichess_1m` появятся:

- `manifest.json`
- `train/shard_*.npz`
- `val/shard_*.npz`

## Обучение

```bash
source .venv/bin/activate
transformer-chess train \
  --dataset-dir data/processed/value_lichess_1m \
  --output artifacts/value_model.pt \
  --epochs 3 \
  --batch-size 256 \
  --precision bf16
```

Для `A100` в Colab используй `--precision bf16`. Значение `auto` тоже выберет `bf16` на CUDA-устройствах с поддержкой `bfloat16`.

## Инференс

Оценить позицию:

```bash
transformer-chess predict-value \
  --checkpoint artifacts/value_model.pt \
  --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
```

Выбрать ход:

```bash
transformer-chess select-move \
  --checkpoint artifacts/value_model.pt \
  --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
```
