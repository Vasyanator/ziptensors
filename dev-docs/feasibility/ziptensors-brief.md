# Сжатие весов моделей без потерь: исследование и бриф для реализации

Исходный бриф перед реализацией. Собраны контекст задачи, результаты замеров, существующие решения и предлагаемый дизайн формата.

## 1. Контекст и требования

- Программа поставляет пользователю **много мелких моделей**, которые запускаются локально. Вместе они занимают заметное место на диске.
- **Код инференса свой.** Распаковку можно встроить в загрузку моделей.
- **Только без потерь.** Никакого квантования, восстановление бит в бит.
- Идея на будущее: оформить подход как открытый формат (рабочие названия `ziptensors` и `ziponnx`). Это не сжатие файла целиком, а перестановка данных в удобный для сжатия вид, затем энтропийное сжатие.

**Неизвестно и нужно уточнить у пользователя:** dtype весов (FP32 / FP16 / BF16), формат моделей (ONNX / safetensors / свой), язык инференса (Python / C++ / другое).

## 2. Главные выводы

1. **Byte shuffle + zstd** — лучший баланс для FP32: в среднем −21% размера весов против −14% у обычного zstd. Распаковка в C (Blosc2) идёт на 1–2 ГБ/с, то есть мелкая модель грузится за миллисекунды.
2. **Потолок для FP32 без потерь около 20–25%.** Два младших байта каждого float имеют энтропию 8,00 бит на байт, то есть это шум, который не сжимается. Сжимается в основном старший байт (знак и экспонента): около 3 бит информации вместо 8.
3. **xz/LZMA вместо zstd** даёт меньше +1 п.п., а распаковывается заметно медленнее. Не стоит того.
4. **Перестановка битов**, чтобы 8-битная экспонента занимала ровно один байт, ничего не даёт: 19,2% против 19,1%.
5. **Общий поток для независимых моделей** не даёт ничего (19,2% против 19,1%). Для моделей с общими весами выигрыш огромный: 5 вариантов Silero VAD вместе ужались на 38% вместо 21%. Значит, нужна **дедупликация тензоров**.
6. **Две проверки, которые могут дать больше 20%:**
   - не хранятся ли в FP32 значения, точно представимые в FP16 или BF16 (обучение в смешанной точности). Если доля таких значений 100%, хранить в FP16 и расширять при загрузке: бит в бит, размер −50%;
   - есть ли общие или почти общие веса между моделями (общий backbone, дообученные версии). Тогда нужны дедупликация или дельта к базовой модели.
7. Для **BF16** ожидаемый выигрыш около 30–33% (экспонента занимает половину бит). Для **FP16** меньше, чем для FP32 (экспонента всего 5 бит). **FP16 и BF16 в этом исследовании не замерялись.**

## 3. Замеры

### Методика

- Модели: настоящие небольшие модели из pip-пакетов со встроенными весами: `rapidocr-onnxruntime==1.2.3` (PP-OCRv3 det/rec, ppocr mobile cls) и `silero-vad==6.2.3` (6 ONNX-вариантов). MTCNN из `facenet-pytorch` пропущен: старый pickle-формат `.pt`.
- Из ONNX извлекались все float-тензоры размером от 16 элементов: initializers, атрибуты узлов `Constant` и вложенные подграфы (`If`/`Loop`). Это важно: у части моделей веса лежат не в `graph.initializer`.
- Сжимались только веса, объединённые в один буфер на модель. Граф не учитывался.
- Компрессоры: `zstandard` уровень 19, `lzma` preset 9 + EXTREME, `blosc2` (ZSTD, фильтры SHUFFLE и BITSHUFFLE).
- Все варианты проверены на побитовое восстановление.
- Все веса в этих моделях оказались FP32, включая `silero_vad_half.onnx`.

### Результаты по моделям (FP32, экономия места; больше — лучше)

| Модель | МБ весов | zstd | xz | shuffle+zstd | split+zstd | shuffle+xz | Энтропия байтов (мл.→ст.), бит | Доля, точно представимая в BF16 / FP16 |
|---|---|---|---|---|---|---|---|---|
| ch_PP-OCRv3_det_infer | 2,37 | 6,9% | 7,5% | 15,4% | 15,6% | 15,6% | 8.00 8.00 7.95 3.02 | 0% / 0% |
| ch_PP-OCRv3_rec_infer | 10,61 | 11,4% | 13,7% | 20,1% | 20,2% | 20,4% | 8.00 8.00 7.16 3.46 | 0% / 0% |
| ch_ppocr_mobile_v2.0_cls | 0,53 | 7,1% | 7,7% | 15,2% | 16,3% | 15,9% | 8.00 8.00 7.97 2.81 | 0% / 0% |
| silero_vad | 2,18 | 17,1% | 18,5% | 21,5% | 21,8% | 22,8% | 7.98 7.99 7.96 3.31 | 1% / 1% |
| silero_vad_16k_op15 | 1,24 | 21,4% | 23,2% | 24,3% | 24,6% | 26,1% | 7.98 7.99 7.96 3.34 | 1% / 1% |
| silero_vad_16k_sequence | 1,24 | 21,4% | 23,2% | 24,3% | 24,6% | 26,1% | 7.98 7.99 7.96 3.34 | 1% / 1% |
| silero_vad_half | 1,24 | 21,5% | 23,3% | 24,5% | 24,8% | 26,2% | 7.98 7.99 7.96 3.25 | 1% / 1% |
| silero_vad_op18_ifless | 2,18 | 17,5% | 18,7% | 21,7% | 21,9% | 23,0% | 7.98 7.99 7.96 3.31 | 1% / 1% |
| silero_vad_openvino_16k | 1,24 | 21,4% | 23,2% | 24,3% | 24,6% | 26,1% | 7.98 7.99 7.96 3.34 | 1% / 1% |
| **Итого (каждая модель отдельно)** | **22,83** | **14,1%** | **15,9%** | **20,7%** | **20,9%** | **21,5%** | | |

- `shuffle+zstd`: байты переставлены (все байты №0, затем все №1 и т. д.), один поток zstd.
- `split+zstd`: каждая байтовая плоскость сжата отдельным потоком zstd.

### Дополнительные замеры

| Эксперимент | Результат |
|---|---|
| Все FP32-веса всех 9 моделей одним потоком (shuffle + zstd 19, long-range, window 2^27) | −38,1%. Выигрыш за счёт одинаковых весов у вариантов Silero |
| Только независимые OCR-модели (13,51 МБ): по отдельности / одним потоком | −19,1% / −19,2%, то есть общий поток не помогает |
| Перестановка битов (rotl 1: экспонента целиком в старшем байте) + shuffle + zstd | −19,2% против −19,1%, то есть бесполезно |
| Blosc2 ZSTD clevel 5, SHUFFLE (все модели) | −19,2%, распаковка 1,95 ГБ/с |
| Blosc2 ZSTD clevel 9, SHUFFLE | −20,4%, распаковка 1,12 ГБ/с |
| Blosc2 ZSTD clevel 5, BITSHUFFLE | −18,4%, распаковка 2,03 ГБ/с |
| Blosc2 ZSTD clevel 9, BITSHUFFLE | −19,6%, распаковка 1,51 ГБ/с |
| Python (zstandard + numpy unshuffle), 1 поток, 10,6 МБ | 0,3–0,4 ГБ/с (узкое место — numpy, а не zstd) |

## 4. Эталонная реализация (проверена, восстанавливает бит в бит для FP32 и FP16)

```python
import numpy as np, zstandard as zstd

def pack(arr, level=19):
    w = arr.dtype.itemsize
    b = np.ascontiguousarray(arr).view(np.uint8).reshape(-1, w).T.tobytes()  # byte shuffle
    return zstd.ZstdCompressor(level=level).compress(b)

def unpack(blob, dtype, shape):
    dtype = np.dtype(dtype)
    b = np.frombuffer(zstd.ZstdDecompressor().decompress(blob), np.uint8)
    return b.reshape(dtype.itemsize, -1).T.copy().view(dtype).reshape(shape)
```

Сжимать каждый тензор отдельно. Имя, dtype и shape хранить в заголовке.

Для C/C++: **c-blosc2** (`BLOSC_SHUFFLE`, `typesize = sizeof(elem)`, кодек ZSTD) делает shuffle и сжатие одним вызовом, с SIMD и многопоточностью. Альтернатива: libzstd плюс свой цикл обратной перестановки байтов.

Проверка FP32 → FP16 без потерь (если результат `1.0`, можно хранить в FP16):

```python
np.mean(w.astype(np.float16).astype(np.float32) == w)
```

Аналог для BF16: `np.mean((w.view(np.uint32) & 0xFFFF) == 0)`.

## 5. Существующие решения

| Проект | Что это | Отличие от задачи |
|---|---|---|
| [ZipNN (IBM)](https://research.ibm.com/blog/Zip-NN-AI-compression), [статья](https://arxiv.org/html/2411.05239v2) | Открытая библиотека для сжатия моделей без потерь, Huffman по экспоненте, плагин для HF, свои сжатые файлы. FP32 −17% (на 8% лучше zstd), BF16 около −33%. Исследует дельта-сжатие похожих моделей | Самый близкий аналог. Проверить, подойдёт ли как готовое решение |
| [Hugging Face Xet](https://huggingface.co/yuchenglow/activity/all) ([код BG4](https://github.com/huggingface/xet-core/blob/main/cas_object/src/byte_grouping/bg4.rs)) | Хаб делает byte grouping (BG4) + LZ4, экономия около 20% | Только хранение и передача. У пользователя на диске файл несжат |
| [zTensor (.zt)](https://docs.rs/crate/ztensor/1.2.3), [репозиторий](https://github.com/pie-project/ztensor), [спецификация](https://github.com/pie-project/ztensor/blob/main/spec/ztensor-v2-spec.md) | Контейнер тензоров: mmap, zstd по тензорам, контрольные суммы, разреженные тензоры, читает safetensors/pt/gguf/npz/onnx/h5 | Byte shuffle в документации не найден. Кандидат на то, чтобы предложить кодирование `byte_group` туда, а не делать свой контейнер |
| HDF5 / Zarr / Blosc2 | Shuffle + zstd для научных массивов, существует давно | Не ориентированы на модели и их загрузчики |
| [compressed-tensors (vLLM)](https://docs.vllm.ai/projects/compressed-tensors) | Расширение safetensors для квантованных и разреженных весов | Сжатие с потерями, не пересекается с задачей |
| DFloat11 | Сжатие BF16 без потерь с распаковкой на GPU во время инференса | Только BF16 и GPU |

## 6. Предлагаемый дизайн

### ziptensors

- **Заголовок в стиле safetensors**: JSON с `name`, `dtype`, `shape`, `offsets`, плюс у каждого тензора поле `encoding`, описанное цепочкой шагов как в Zarr, например `["byte_group", "zstd"]`. Новые шаги (Huffman по экспоненте, дельта) добавляются без смены версии формата.
- **Чанки 1–4 МБ** внутри больших тензоров: параллельная распаковка и чтение отдельных тензоров без распаковки всего файла.
- **Преобразование зависит от dtype**: для FP32, FP16 и BF16 группировка байтов; для INT8/INT4 и квантованных весов только zstd или хранение как есть (они почти не сжимаются).
- **Дедупликация по хешу содержимого**: одинаковый тензор хранится один раз, остальные ссылаются на него (внутри файла и между файлами набора).
- **Дельта к базовой модели** как отдельное кодирование (XOR или вычитание, затем сжатие) для дообученных версий.
- **Безопасность**: размер после распаковки и контрольная сумма несжатых данных обязательно в заголовке (защита от повреждений и zip-бомб); никакого исполняемого кода, как в safetensors.
- **Компромисс**: zero-copy mmap со сжатыми тензорами невозможен. Допустим режим «без сжатия» для отдельных тензоров.

### ziponnx

- Граф ONNX (protobuf без весов) маленький, сжимается обычным zstd.
- Веса выносятся как ONNX external data, но в контейнер ziptensors.
- Загрузка, вариант 1 (модели меньше 2 ГБ): собрать полный ModelProto в памяти и передать байты в `onnxruntime.InferenceSession(bytes)`.
- Загрузка, вариант 2: передать распакованные веса в ONNX Runtime как внешние initializers (`SessionOptions.add_external_initializers` в Python, `AddExternalInitializers` или `AddExternalInitializersFromFilesInMemory` в C API; сигнатуры проверить по документации нужной версии ORT).
- По сути ziponnx = граф + ziptensors, а не отдельный формат.

### Позиционирование

- Как внутренний формат программы — однозначно делать.
- Как открытый стандарт ниша реальна (приложения и игры, поставляющие набор моделей пользователю), но массово скачиваемые модели уже сжимаются при передаче через Xet. Без поддержки в onnxruntime, llama.cpp или transformers формат останется нишевым. Альтернатива: предложить кодирование в zTensor или держать совместимость с заголовком safetensors.
- Название с префиксом «zip» вводит в заблуждение: `.pt` и `.npz` уже являются ZIP-архивами, а ZipNN использует похожее имя.

## 7. Рекомендуемые следующие шаги

1. Прогнать `bench.py` (приложение A) на реальных моделях пользователя. Посмотреть долю значений, точно представимых в FP16/BF16, и повторяющиеся тензоры между моделями.
2. Оценить ZipNN и zTensor как готовые решения, прежде чем писать своё.
3. Если своё: спецификация v0.1 (заголовок, цепочки кодирований, чанки, дедупликация, контрольные суммы), эталонная реализация на Python, затем на языке инференса (C++ через c-blosc2 или libzstd).
4. Тесты: побитовое восстановление для всех dtype, повреждённые и злонамеренные файлы, замер времени загрузки против несжатого файла на SSD и HDD.

## Приложение A. Скрипт замеров (`bench.py`)

Зависимости: `pip install numpy zstandard onnx`. Запуск: `python3 -I bench.py <папка_с_моделями>` (ищет рекурсивно `*.onnx` и `*.pt`).

```python
"""Lossless compression benchmark for model weights.

Usage: python3 -I bench.py <models_root>
"""
import sys, os, glob, time, lzma, zipfile, io
import numpy as np
import zstandard as zstd
import onnx
from onnx import numpy_helper

root = sys.argv[1]


def onnx_weights(path):
    m = onnx.load(path)
    out = {}

    def add(t):
        a = numpy_helper.to_array(t)
        if a.dtype in (np.float32, np.float16) and a.size >= 16:
            out.setdefault(a.dtype, []).append(a.ravel())

    def walk(g):
        for t in g.initializer:
            add(t)
        for n in g.node:
            for at in n.attribute:
                if at.type == onnx.AttributeProto.TENSOR:
                    add(at.t)
                elif at.type == onnx.AttributeProto.GRAPH:
                    walk(at.g)
                elif at.type == onnx.AttributeProto.GRAPHS:
                    for sg in at.graphs:
                        walk(sg)

    walk(m.graph)
    return {k: np.concatenate(v) for k, v in out.items()}


def pt_weights(path):
    # torch .pt zip: raw storages live in */data/<n>; assumes fp32
    try:
        z = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        return {}  # legacy pickle format, skip
    arrs = [np.frombuffer(z.read(n), dtype=np.float32)
            for n in z.namelist() if "/data/" in n and not n.endswith(".pkl")]
    return {np.dtype(np.float32): np.concatenate(arrs)} if arrs else {}


def shuffle(b: bytes, width: int) -> bytes:
    a = np.frombuffer(b, dtype=np.uint8).reshape(-1, width)
    return a.T.copy().tobytes()


def split_streams(b: bytes, width: int):
    a = np.frombuffer(b, dtype=np.uint8).reshape(-1, width)
    return [a[:, i].copy().tobytes() for i in range(width)]


def entropy_bits(stream: bytes) -> float:
    c = np.bincount(np.frombuffer(stream, dtype=np.uint8), minlength=256).astype(float)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())


Z19 = zstd.ZstdCompressor(level=19)
D = zstd.ZstdDecompressor()


def xz(b):
    return lzma.compress(b, preset=9 | lzma.PRESET_EXTREME)


files = sorted(glob.glob(os.path.join(root, "**/*.onnx"), recursive=True) +
               glob.glob(os.path.join(root, "**/*.pt"), recursive=True))

rows = []
all_fp32 = []
tot = dict(raw=0, z_raw=0, xz_raw=0, z_shuf=0, z_split=0, xz_shuf=0)

for f in files:
    ws = onnx_weights(f) if f.endswith(".onnx") else pt_weights(f)
    for dt, arr in ws.items():
        b = arr.tobytes()
        w = arr.dtype.itemsize
        raw = len(b)
        z_raw = len(Z19.compress(b))
        xz_raw = len(xz(b))
        sh = shuffle(b, w)
        z_shuf = len(Z19.compress(sh))
        streams = split_streams(b, w)
        z_split = sum(len(Z19.compress(s)) for s in streams)
        xz_shuf = len(xz(sh))
        ent = [entropy_bits(s) for s in streams]  # little-endian: last = sign+exp hi
        extra = ""
        if arr.dtype == np.float32:
            u = arr.view(np.uint32)
            bf16_exact = float(np.mean((u & 0xFFFF) == 0))
            fp16_exact = float(np.mean(arr.astype(np.float16).astype(np.float32) == arr))
            extra = f"bf16-exact={bf16_exact:.0%} fp16-exact={fp16_exact:.0%}"
            all_fp32.append(arr)
        name = os.path.basename(f)
        rows.append((name, str(arr.dtype), raw, z_raw, xz_raw, z_shuf, z_split, xz_shuf, ent, extra))
        for k, v in zip(tot, (raw, z_raw, xz_raw, z_shuf, z_split, xz_shuf)):
            tot[k] += v

pct = lambda c, r: f"{100 * (1 - c / r):5.1f}%"
print(f"{'model':38} {'dtype':7} {'MB':>6} | saved: {'zstd':>6} {'xz':>6} {'shuf+zstd':>9} {'split+zstd':>10} {'shuf+xz':>8} | byte entropy (bits, lo..hi)")
for name, dt, raw, a, b2, c, d, e, ent, extra in rows:
    print(f"{name[:38]:38} {dt:7} {raw/1e6:6.2f} | {pct(a,raw):>12} {pct(b2,raw):>6} {pct(c,raw):>9} {pct(d,raw):>10} {pct(e,raw):>8} | "
          + " ".join(f"{x:.2f}" for x in ent) + ("  " + extra if extra else ""))
r = tot["raw"]
print(f"{'TOTAL (each model separately)':38} {'':7} {r/1e6:6.2f} | {pct(tot['z_raw'],r):>12} {pct(tot['xz_raw'],r):>6} {pct(tot['z_shuf'],r):>9} {pct(tot['z_split'],r):>10} {pct(tot['xz_shuf'],r):>8}")

# Solid stream: all fp32 weights of all models together (cross-model redundancy)
if all_fp32:
    big = np.concatenate(all_fp32).tobytes()
    zl = zstd.ZstdCompressor(compression_params=zstd.ZstdCompressionParameters.from_level(
        19, window_log=27, enable_ldm=True))
    solid = len(zl.compress(shuffle(big, 4)))
    print(f"\nAll fp32 weights in one solid stream (shuffle + zstd19 long): {len(big)/1e6:.2f} MB -> saved {pct(solid, len(big))}")

    # Decompression speed for shuffle+zstd on the largest fp32 tensor set
    largest = max(all_fp32, key=len).tobytes()
    comp = Z19.compress(shuffle(largest, 4))
    t = time.perf_counter()
    N = 20
    for _ in range(N):
        d = D.decompress(comp)
        a = np.frombuffer(d, dtype=np.uint8).reshape(4, -1).T.copy()  # unshuffle
    t = (time.perf_counter() - t) / N
    assert a.tobytes() == largest
    print(f"Decompress+unshuffle speed (1 thread, python): {len(largest)/t/1e9:.2f} GB/s on {len(largest)/1e6:.1f} MB; roundtrip bit-exact: OK")
```
