import time
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MnistCNN


# Основные параметры обучения.
BATCH_SIZE = 128
EPOCHS = 120
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

# Папка датасета и файл для сохранения обученных весов.
DATA_DIR = Path(__file__).parent / "data"
WEIGHTS_PATH = Path(__file__).parent / "model_weights.pth"


def get_dataloaders() -> tuple[DataLoader, DataLoader]:
    # Для обучения добавляются случайные искажения, чтобы модель лучше распознавала разные почерки.
    train_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=15,
            translate=(0.15, 0.15),
            scale=(0.7, 1.3),
            shear=10,
            fill=0,
        ),
        transforms.ToTensor(),
    ])
    # Тестовые данные оставляем без искажений.
    test_transform = transforms.ToTensor()

    # torchvision сам скачает MNIST, если файлов ещё нет в папке data.
    train_set = datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=train_transform
    )
    test_set = datasets.MNIST(
        root=DATA_DIR, train=False, download=True, transform=test_transform
    )

    # DataLoader разбивает датасет на батчи для обучения.
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, loss_fn, device) -> float:
    # Режим обучения включает Dropout и обновление статистики BatchNorm.
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        # Переносим батч на CPU или GPU.
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Один шаг обучения: предсказание, ошибка, градиенты, обновление весов.
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        # Ошибка накапливается с учётом размера текущего батча.
        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device) -> float:
    # Режим проверки отключает Dropout.
    model.eval()
    correct = 0
    total = 0

    # На проверке веса не меняются, поэтому градиенты не считаются.
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    # Возвращаем долю правильных ответов.
    return correct / total


def format_duration(seconds: float) -> str:
    # Формат времени используется для вывода ETA в консоль.
    seconds = int(seconds)
    if seconds >= 3600:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h:d}:{m:02d}:{s:02d}"
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"


def main() -> None:
    # Если доступна CUDA, обучение идёт на видеокарте.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство для обучения: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    train_loader, test_loader = get_dataloaders()
    print(f"Картинок в обучающей выборке: {len(train_loader.dataset)}")
    print(f"Картинок в тестовой выборке:  {len(test_loader.dataset)}")
    print(f"Эпох: {EPOCHS}, размер батча: {BATCH_SIZE}, lr: {LEARNING_RATE}")

    # Создаём модель, оптимизатор, планировщик learning rate и функцию ошибки.
    model = MnistCNN().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()

    print("\nНачинаю обучение...\n")
    best_accuracy = 0.0
    start_time = time.time()

    # Основной цикл обучения по эпохам.
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        accuracy = evaluate(model, test_loader, device)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # Сохраняем веса только при новой лучшей точности.
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), WEIGHTS_PATH)
            tag = "*save"
        else:
            tag = "     "

        elapsed = time.time() - start_time
        eta = elapsed / epoch * (EPOCHS - epoch)

        # Выводим прогресс обучения после каждой эпохи.
        print(
            f"Эпоха {epoch:3d}/{EPOCHS} | "
            f"loss: {avg_loss:.4f} | "
            f"accuracy: {accuracy * 100:5.2f}% | "
            f"best: {best_accuracy * 100:5.2f}% | "
            f"lr: {current_lr:.5f} | "
            f"ETA: {format_duration(eta)} | "
            f"{tag}"
        )

    total_time = time.time() - start_time
    print(f"\nГотово за {format_duration(total_time)}.")
    print(f"Лучшая точность на тесте: {best_accuracy * 100:.2f}%")
    print(f"Веса сохранены в: {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
