# 🚗 Детектирование и распознавание автомобильных номеров

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Object%20Detection-orange.svg)
![Tesseract](https://img.shields.io/badge/Tesseract-OCR-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Автоматическое обнаружение и распознавание номерных знаков автомобилей с использованием глубокого обучения**

[Возможности](#-возможности) • [Установка](#-установка) • [Использование](#-использование) • [Технологии](#-технологии)

</div>

---

## 📋 Описание

Проект представляет собой систему автоматического детектирования и распознавания автомобильных номеров на изображениях и видео. Система использует модель глубокого обучения **SSD Inception V2** для обнаружения номерных знаков и **Tesseract OCR** для распознавания текста.

### ✨ Особенности

- 🎯 **Высокая точность детектирования** - модель обучена на датасете из ~3000 изображений российских номеров
- 📸 **Работа с изображениями** - обработка статических изображений
- 🎥 **Видео в реальном времени** - детектирование номеров в видеопотоке
- 🔍 **Визуализация результатов** - отображение bounding box'ов вокруг обнаруженных номеров
- 🌍 **Поддержка русских номеров** - оптимизировано для распознавания номеров РФ

---

## 🛠 Технологии

- **TensorFlow Object Detection API** - фреймворк для работы с моделями детектирования объектов
- **SSD Inception V2** - нейросетевая модель для обнаружения номерных знаков
- **Tesseract OCR** - open-source движок для оптического распознавания символов
- **OpenCV** - обработка изображений и видео
- **Python 3.x** - основной язык программирования

---

## 📦 Установка

### Предварительные требования

- Python 3.7+
- TensorFlow
- Tesseract OCR (с поддержкой русского языка)
- TensorFlow Object Detection API

### Пошаговая установка

#### 1️⃣ Установка TensorFlow Object Detection API

1. Клонируйте репозиторий TensorFlow Models:
```bash
git clone https://github.com/tensorflow/models.git
```

2. Добавьте следующие пути в переменную окружения `PYTHONPATH`:

**Для Windows:**
```powershell
[директория tensorflow models]\research
[директория tensorflow models]\research\slim
[директория tensorflow models]\research\nets
[директория tensorflow models]\research\object_detection
[директория tensorflow models]\research\object_detection\utils
```

**Для Linux:**
```bash
export PYTHONPATH=$PYTHONPATH:[директория tensorflow models]/research
export PYTHONPATH=$PYTHONPATH:[директория tensorflow models]/research/slim
export PYTHONPATH=$PYTHONPATH:[директория tensorflow models]/research/nets
export PYTHONPATH=$PYTHONPATH:[директория tensorflow models]/research/object_detection
export PYTHONPATH=$PYTHONPATH:[директория tensorflow models]/research/object_detection/utils
```

#### 2️⃣ Установка Tesseract OCR

**Windows:**
- Скачайте установщик с [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- ⚠️ **Важно:** При установке выберите русский язык (Russian)

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr-rus
```

**Linux (другие дистрибутивы):**
- Скачайте с [официального репозитория](https://github.com/tesseract-ocr/tesseract/releases)

#### 3️⃣ Установка Python-зависимостей

```bash
# Основные библиотеки
pip install tensorflow opencv-python pytesseract numpy matplotlib cython path.py

# PyCOCOtools (для TensorFlow Models)
```

**Установка PyCOCOtools:**

**Windows:**
1. Убедитесь, что у вас установлены **Visual C++ 2015 Build Tools** ([скачать](https://go.microsoft.com/fwlink/?LinkId=691126))
2. Добавьте их в переменную окружения `PATH`
3. Установите:
```bash
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```

**Linux:**
```bash
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
```

#### 4️⃣ Компиляция Protobuf файлов

**Linux:**
```bash
# Установка protoc compiler
sudo curl -OL https://github.com/google/protobuf/releases/download/v3.6.1/protoc-3.6.1-linux-x86_64.zip
unzip protoc-3.6.1-linux-x86_64.zip -d protoc3
sudo mv protoc3/bin/* /usr/local/bin/
sudo mv protoc3/include/* /usr/local/include/

# Компиляция .proto файлов
cd tensorflow/models/research
protoc object_detection/protos/*.proto --python_out=.
```

**Windows:**
1. Скачайте [NuGet](https://www.nuget.org/downloads)
2. Добавьте `nuget.exe` в переменную окружения `PATH`
3. Установите Google.Protobuf.Tools:
```powershell
nuget.exe install Google.Protobuf.Tools
```
4. Добавьте путь к `protoc.exe` в `PATH` (например: `D:\Nuget\Google.Protobuf.Tools.3.4.0\tools\windows_x64`)
5. Компилируйте:
```powershell
cd tensorflow\models\research
protoc object_detection\protos\*.proto --python_out=.
```

---

## 🚀 Использование

### Режим работы с изображениями

```bash
python car_number_detector_ssd.py -m 0 -i path/to/image.jpg
```

**Параметры:**
- `-m 0` - режим детектирования на изображении
- `-i` - путь к изображению

**Результат:**
- Изображение с отмеченными bounding box'ами вокруг обнаруженных номеров
- Список распознанных номеров в консоли

### Режим работы с видео

```bash
python car_number_detector_ssd.py -m 1
```

**Параметры:**
- `-m 1` - режим видео-детектирования в реальном времени

**Результат:**
- Окно с видео в реальном времени и визуализацией детектирования
- Список распознанных номеров в консоли
- Нажмите `q` для выхода

---

## 📊 Входные и выходные данные

### Входные данные

| Параметр | Описание |
|----------|----------|
| `-m, --mode` | Режим работы: `0` - изображение, `1` - видео |
| `-i, --image_path` | Путь к изображению (для режима 0) |

### Выходные данные

**Режим изображения:**
- Визуализированное изображение с bounding box'ами
- Список распознанных номеров в консоли

**Режим видео:**
- Окно с видео в реальном времени
- Список распознанных номеров в консоли

---

## 🔧 Архитектура решения

1. **Детектирование** - модель SSD Inception V2 обнаруживает номерные знаки на изображении/кадре
2. **Обрезка** - извлекаются области с обнаруженными номерами
3. **Предобработка** - улучшение качества изображения для OCR:
   - Увеличение разрешения (2x)
   - Билинейная фильтрация
   - Конвертация в градации серого
4. **Распознавание** - Tesseract OCR извлекает текст из обработанного изображения

---

## 💡 Возможные улучшения

- [ ] Улучшение OCR функции (замена Tesseract на более точные модели)
- [ ] Улучшение предобработки изображения перед OCR
- [ ] Выравнивание номера перед подачей на OCR
- [ ] Удаление искажений и перспективных преобразований
- [ ] Поддержка других типов номеров (не только РФ)
- [ ] Добавление веб-интерфейса
- [ ] Оптимизация для работы на GPU

---

## 🔄 Альтернативные подходы

### Рассмотренные, но не использованные решения:

1. **Детектор граней OpenCV** - не работает при бликах и загрязнениях
2. **Каскады Хаара** - много ложных срабатываний
3. **Кастомный OCR** - требует дополнительной разработки и обучения

---

---

Автор: legenda  
Telegram: [@LEGENDA_SD](https://t.me/LEGENDA_SD)
