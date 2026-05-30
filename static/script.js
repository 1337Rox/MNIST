const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

// Элементы управления и вывода результата.
const predictBtn = document.getElementById("predict-btn");
const clearBtn = document.getElementById("clear-btn");
const resultBlock = document.getElementById("result");
const resultDigit = document.getElementById("result-digit");
const resultConfidence = document.getElementById("result-confidence");
const probabilitiesList = document.getElementById("probabilities");
const hint = document.getElementById("hint");
const defaultHintText = hint.textContent;

// Настройки кисти для рисования на canvas.
const STROKE_WIDTH = 18;
const STROKE_COLOR = "#000";

ctx.lineWidth = STROKE_WIDTH;
ctx.strokeStyle = STROKE_COLOR;
ctx.lineCap = "round";
ctx.lineJoin = "round";

// Состояние рисования на холсте.
let isDrawing = false;
let hasDrawnAnything = false;


function getPointerPosition(event) {
    // Координаты курсора переводятся в координаты внутри canvas.
    const rect = canvas.getBoundingClientRect();
    const point = event.touches ? event.touches[0] : event;
    return {
        x: point.clientX - rect.left,
        y: point.clientY - rect.top,
    };
}


function startDrawing(event) {
    // Начинаем новую линию при нажатии мыши или касании.
    event.preventDefault();
    isDrawing = true;
    hasDrawnAnything = true;
    const { x, y } = getPointerPosition(event);
    ctx.beginPath();
    ctx.moveTo(x, y);
}


function draw(event) {
    // Пока пользователь удерживает мышь или палец, линия продолжается.
    if (!isDrawing) return;
    event.preventDefault();
    const { x, y } = getPointerPosition(event);
    ctx.lineTo(x, y);
    ctx.stroke();
}


function stopDrawing() {
    // Завершаем рисование, когда пользователь отпустил мышь или палец.
    isDrawing = false;
}


// Одни и те же функции используются для мыши и touch-событий.
canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mouseleave", stopDrawing);
canvas.addEventListener("touchstart", startDrawing);
canvas.addEventListener("touchmove", draw);
canvas.addEventListener("touchend", stopDrawing);
canvas.addEventListener("touchcancel", stopDrawing);


function clearCanvas() {
    // Очистка сбрасывает холст, результат и ошибку.
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    hasDrawnAnything = false;
    resultBlock.hidden = true;
    hint.textContent = defaultHintText;
    hint.classList.remove("error");
}


clearBtn.addEventListener("click", clearCanvas);


async function predict() {
    // Без рисунка запрос на сервер не отправляется.
    if (!hasDrawnAnything) {
        showError("Сначала нарисуйте цифру.");
        return;
    }

    // Сервер принимает PNG-картинку как base64-строку.
    const imageData = canvas.toDataURL("image/png");

    predictBtn.disabled = true;
    predictBtn.textContent = "Распознаю...";

    try {
        // Отправляем изображение на Flask endpoint /predict.
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: imageData }),
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.error || `Ошибка сервера: ${response.status}`);
        }

        const data = await response.json();
        showResult(data);
    } catch (err) {
        // Ошибки сервера или сети показываются под canvas.
        showError(err.message || "Не удалось связаться с сервером.");
    } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = "Распознать";
    }
}


predictBtn.addEventListener("click", predict);


function showResult(data) {
    // На странице показывается итоговая цифра и вероятности по всем классам.
    hint.textContent = defaultHintText;
    hint.classList.remove("error");

    resultDigit.textContent = data.digit;
    resultConfidence.textContent =
        `Уверенность: ${(data.confidence * 100).toFixed(1)}%`;

    // Для каждой цифры строится отдельная строка с полосой вероятности.
    probabilitiesList.innerHTML = "";
    data.probabilities.forEach((prob, digit) => {
        const li = document.createElement("li");
        if (digit === data.digit) {
            li.classList.add("top");
        }

        const digitSpan = document.createElement("span");
        digitSpan.className = "prob-digit";
        digitSpan.textContent = digit;

        const bar = document.createElement("div");
        bar.className = "prob-bar";
        const fill = document.createElement("div");
        fill.className = "prob-bar-fill";
        fill.style.width = `${(prob * 100).toFixed(1)}%`;
        bar.appendChild(fill);

        const valueSpan = document.createElement("span");
        valueSpan.className = "prob-value";
        valueSpan.textContent = `${(prob * 100).toFixed(1)}%`;

        li.appendChild(digitSpan);
        li.appendChild(bar);
        li.appendChild(valueSpan);
        probabilitiesList.appendChild(li);
    });

    resultBlock.hidden = false;
}


function showError(message) {
    // Ошибка скрывает старый результат, чтобы не путать пользователя.
    hint.textContent = message;
    hint.classList.add("error");
    resultBlock.hidden = true;
}
