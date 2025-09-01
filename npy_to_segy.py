# import numpy as np
# import segyio
#
# def npy_to_segy(npy_path: str,
#                 segy_path: str,
#                 dt: float = 0.004,
#                 sample_format: int = 1) -> None:
#     """
# Преобразует .npy файл с данными сейсмических трасс в формат SEG-Y.
#
# Parameters:
# -------------
# npy_path : str
# Путь к .npy файлу. Ожидается массив формы (n_traces, n_samples).
# segy_path : str
# Путь для сохранения выходного SEG-Y файла.
# dt : float, default=0.004
# Интервал дискретизации (sampling interval) в секундах.
# sample_format : int, default=1
# Формат сэмплов (1 — IBM плавающая точка, 5 — IEEE плавающая точка и т.д.).
#     """
#     # Загружаем данные из .npy
#     data = np.load(npy_path)
#
#     # Проверяем форму массива
#     if data.ndim != 2:
#         raise ValueError(f"Ожидается 2D массив, получено {data.ndim}D массив")
#     n_traces, n_samples = data.shape
#
#     # Настраиваем спецификацию SEG-Y
#     spec = segyio.spec()
#     spec.tracecount = n_traces
#     spec.samples = np.arange(n_samples) * dt
#     spec.format = sample_format
#
#     # Создаем SEG-Y файл
#     with segyio.create(segy_path, spec) as segy:
#         # Заполняем текстовый заголовок
#         text = f"Converted from {npy_path} by npy_to_segy"
#         segy.text[0] = segyio.tools.wrap(text)
#
#         # Записываем каждую трассу
#         for tr in range(n_traces):
#             segy.trace[tr] = data[tr, :]
#
#     print(f"Успешно сохранено {n_traces} трасс в {segy_path}")
#
#
# if __name__ == '__main__':
#     import argparse
#
#     parser = argparse.ArgumentParser(description="Convert .npy seismic data to .segy format")
#     parser.add_argument("npy_file", help="Path to the input .npy file")
#     parser.add_argument("segy_file", help="Path to the output .segy file")
#     parser.add_argument("--dt", type=float, default=0.004,
#                         help="Sampling interval in seconds (default: 0.004)")
#     parser.add_argument("--format", type=int, default=1,
#                         help="Sample format code (1=IBM float, 5=IEEE float, etc.)")
#     args = parser.parse_args()
#
#     npy_to_segy(args.npy_file, args.segy_file, args.dt, args.format)
#

# import numpy as np
# import segyio
# import os
# from pathlib import Path
#
# def npy_to_segy(npy_path: str,
#                 segy_path: str,
#                 dt: float = 0.004,
#                 sample_format: int = 1) -> None:
#     """
# Преобразует .npy файл с данными сейсмических трасс в формат SEG-Y.
#
# Parameters:
# -------------
# npy_path : str
# Путь к .npy файлу. Ожидается массив формы (n_traces, n_samples).
# segy_path : str
# Путь для сохранения выходного SEG-Y файла.
# dt : float, default=0.004
# Интервал дискретизации (sampling interval) в секундах.
# sample_format : int, default=1
# Формат сэмплов (1 — IBM плавающая точка, 5 — IEEE плавающая точка и т.д.).
#     """
#     # Загружаем данные из .npy
#     data = np.load(npy_path)
#
#     # Проверяем форму массива
#     if data.ndim != 2:
#         raise ValueError(f"Ожидается 2D массив, получено {data.ndim}D массив")
#     n_traces, n_samples = data.shape
#
#     # Настраиваем спецификацию SEG-Y
#     spec = segyio.spec()
#     spec.tracecount = n_traces
#     spec.samples = np.arange(n_samples) * dt
#     spec.format = sample_format
#
#     # Создаем SEG-Y файл
#     with segyio.create(segy_path, spec) as segy:
#         # Заполняем текстовый заголовок
#         text = f"Converted from {os.path.basename(npy_path)} by npy_to_segy"
#         segy.text[0] = segyio.tools.wrap(text)
#
#         # Записываем каждую трассу
#         for tr in range(n_traces):
#             segy.trace[tr] = data[tr, :]
#
#     print(f"Успешно сохранен {os.path.basename(segy_path)} ({n_traces} трасс)")
#
#
# def convert_directory(input_dir: Path,
#                       output_dir: Path,
#                       dt: float,
#                       sample_format: int) -> None:
#     """
# Обходит все .npy файлы в директории input_dir и конвертирует их в SEG-Y в output_dir.
#     """
#     if not input_dir.exists() or not input_dir.is_dir():
#         raise ValueError(f"Входная папка не найдена: {input_dir}")
#     output_dir.mkdir(parents=True, exist_ok=True)
#
#     npy_files = list(input_dir.glob('*.npy'))
#     if not npy_files:
#         print(f"Нет файлов .npy в папке {input_dir}")
#         return
#
#     for npy_file in npy_files:
#         segy_file = output_dir / f"{npy_file.stem}.segy"
#         npy_to_segy(str(npy_file), str(segy_file), dt, sample_format)
#
#     print(f"Конвертация завершена. Файлы сохранены в {output_dir}")
#
#
# if __name__ == '__main__':
#     import argparse
#
#     parser = argparse.ArgumentParser(
#         description="Конвертация всех .npy файлов в папке в формат SEG-Y")
#     parser.add_argument(
#         'input_folder',
#         help="Путь к папке с .npy файлами"
#     )
#     parser.add_argument(
#         '--output_folder',
#         help="Путь к выходной папке (по умолчанию: input_folder_segy)",
#         default=None
#     )
#     parser.add_argument(
#         '--dt',
#         type=float,
#         default=0.004,
#         help="Интервал дискретизации в секундах (default: 0.004)"
#     )
#     parser.add_argument(
#         '--format',
#         type=int,
#         default=1,
#         help="Код формата сэмпла (1=IBM float, 5=IEEE float, etc.)"
#     )
#     args = parser.parse_args()
#
#     # input_path = Path(args.input_folder).resolve()
#     input_path = r'.\npy_data'
#     if args.output_folder:
#         output_path = Path(args.output_folder).resolve()
#     else:
#         parent = input_path.parent
#         # output_path = parent / f"{input_path.name}_segy"
#         output_path = r'.\segy_data'
#
#
#     convert_directory(input_path, output_path, args.dt, args.format)

import numpy as np
import segyio
import os
from pathlib import Path

def npy_to_segy(npy_path: str,
                segy_path: str,
                dt: float = 0.004,
                sample_format: int = 1) -> None:
    """
Преобразует .npy файл с данными сейсмических трасс в формат SEG-Y.

Parameters:
-------------
npy_path : str
Путь к .npy файлу. Ожидается массив формы (n_traces, n_samples).
segy_path : str
Путь для сохранения выходного SEG-Y файла.
dt : float, default=0.004
Интервал дискретизации (sampling interval) в секундах.
sample_format : int, default=1
Формат сэмплов (1 — IBM плавающая точка, 5 — IEEE плавающая точка и т.д.).
    """
    data = np.load(npy_path)

    if data.ndim != 2:
        raise ValueError(f"Ожидается 2D массив, получено {data.ndim}D массив")
    n_traces, n_samples = data.shape

    spec = segyio.spec()
    spec.tracecount = n_traces
    spec.samples = np.arange(n_samples) * dt
    spec.format = sample_format

    with segyio.create(segy_path, spec) as segy:
        text = f"Converted from {os.path.basename(npy_path)} by npy_to_segy"
        segy.text[0] = segyio.tools.wrap(text)

        for tr in range(n_traces):
            segy.trace[tr] = data[tr, :]

    print(f"Успешно сохранен {os.path.basename(segy_path)} ({n_traces} трасс)")


def convert_directory(input_dir: Path,
                      output_dir: Path,
                      dt: float,
                      sample_format: int) -> None:
    """
Обходит все .npy файлы в директории input_dir и конвертирует их в SEG-Y в output_dir.
    """
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Входная папка не найдена: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_files = list(input_dir.glob('*.npy'))
    if not npy_files:
        print(f"Нет файлов .npy в папке {input_dir}")
        return

    for npy_file in npy_files:
        segy_file = output_dir / f"{npy_file.stem}.segy"
        npy_to_segy(str(npy_file), str(segy_file), dt, sample_format)

    print(f"Конвертация завершена. Файлы сохранены в {output_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Конвертация всех .npy файлов в папке в формат SEG-Y"
    )
    parser.add_argument(
        '--input_folder',
        help="Путь к папке с .npy файлами",
        default=None
    )
    parser.add_argument(
        '--output_folder',
        help="Путь к выходной папке (по умолчанию: segy_data)",
        default=None
    )
    parser.add_argument(
        '--dt',
        type=float,
        default=0.004,
        help="Интервал дискретизации в секундах (default: 0.004)"
    )
    parser.add_argument(
        '--format',
        type=int,
        default=1,
        help="Код формата сэмпла (1=IBM float, 5=IEEE float, etc.)"
    )
    args = parser.parse_args()

    # Определение базовой директории (где находится скрипт)
    base_dir = Path(__file__).parent.resolve()

    # Установка путей к папкам .npy и .segy
    if args.input_folder:
        input_path = Path(args.input_folder).resolve()
    else:
        input_path = base_dir / 'data'

    if args.output_folder:
        output_path = Path(args.output_folder).resolve()
    else:
        output_path = base_dir / 'segy'

    convert_directory(input_path, output_path, args.dt, args.format)
    
