import base64
import io
from pathlib import Path

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps
from torchvision import transforms

from model import MnistCNN


# Пути и устройство для работы модели.
# Файл с весами, которые создаёт train.py.
WEIGHTS_PATH = Path(__file__).parent / "model_weights.pth"

if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(
        f"Не найден файл с весами: {WEIGHTS_PATH}\n"
        "Сначала обучите модель командой: python train.py"
    )

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Создание и настройка модели для предсказаний.
# Модель загружается один раз при старте сервера.
model = MnistCNN().to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.eval()

# ToTensor переводит PIL-картинку в тензор PyTorch.
to_tensor = transforms.ToTensor()

# Flask-приложение, которое отдаёт страницу и принимает запросы.
app = Flask(__name__)


def preprocess_canvas_image(data_url: str) -> torch.Tensor:
    # Функция готовит рисунок пользователя к тому же виду, что и картинки MNIST.
    # Canvas отправляет изображение как data URL, внутри которого лежит base64.
    _, base64_data = data_url.split(",", maxsplit=1)
    image = Image.open(io.BytesIO(base64.b64decode(base64_data)))

    # Приводим рисунок к формату MNIST: серый цвет, инверсия, размер 28x28.
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    image = Image.alpha_composite(background, image.convert("RGBA")).convert("L")
    image = ImageOps.invert(image)

    bbox = image.getbbox()
    if bbox is None:
        # Если пользователь ничего не нарисовал, отправляем пустое изображение.
        canvas = Image.new("L", (28, 28), 0)
    else:
        # Обрезаем пустые поля и вписываем цифру в центр изображения.
        digit = image.crop(bbox)
        width, height = digit.size

        # Сохраняем пропорции цифры при уменьшении.
        if width > height:
            new_width = 20
            new_height = max(1, round(20 * height / width))
        else:
            new_height = 20
            new_width = max(1, round(20 * width / height))

        digit = digit.resize((new_width, new_height), Image.LANCZOS)

        # Финальная картинка для модели всегда 28x28.
        canvas = Image.new("L", (28, 28), 0)
        offset_x = (28 - new_width) // 2
        offset_y = (28 - new_height) // 2
        canvas.paste(digit, (offset_x, offset_y))

    # Добавляем размерность batch: (1, 1, 28, 28).
    tensor = to_tensor(canvas).unsqueeze(0)
    return tensor.to(DEVICE)


@app.route("/")
def index():
    # Главная страница с холстом для рисования.
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Получаем картинку из браузера и возвращаем результат распознавания.
    payload = request.get_json(silent=True)
    if not payload or "image" not in payload:
        return jsonify({"error": "В запросе нет поля 'image'"}), 400

    try:
        tensor = preprocess_canvas_image(payload["image"])
    except Exception as exc:
        return jsonify({"error": f"Не удалось обработать картинку: {exc}"}), 400

    # Предсказание делается без обучения и без расчёта градиентов.
    with torch.no_grad():
        logits = model(tensor)
        # Softmax превращает выходы модели в вероятности.
        probabilities = F.softmax(logits, dim=1).squeeze(0)

    # Берём цифру с самой большой вероятностью.
    predicted_digit = int(probabilities.argmax().item())
    confidence = float(probabilities[predicted_digit].item())
    probabilities_list = [float(p) for p in probabilities.cpu().tolist()]

    # JSON-ответ использует JavaScript на странице.
    return jsonify(
        {
            "digit": predicted_digit,
            "confidence": confidence,
            "probabilities": probabilities_list,
        }
    )


if __name__ == "__main__":
    # Локальный запуск сервера.
    app.run(debug=True, host="127.0.0.1", port=5000)
