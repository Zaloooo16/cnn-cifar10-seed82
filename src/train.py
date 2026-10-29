"""
Pregunta 4: ciclo de entrenamiento, metricas y curvas de aprendizaje.

Uso (desde la raiz del repositorio):
    python src/train.py

Genera:
    outputs/loss_accuracy.png     curvas de Loss y Accuracy (train vs val)
    outputs/confusion_matrix.png  matriz de confusion sobre el conjunto de prueba
    outputs/metrics.json          metricas numericas
    README.md                     (rellena automaticamente la seccion de resultados)
"""
import json
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix

from dataset import CLASSES, OUTPUT_DIR, ROOT_DIR, SEED, get_dataloaders, set_seed
from model import CNN, count_parameters

# ------------------------- Hiperparametros -------------------------
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, criterion, optimizer=None):
    """Una pasada completa. Si optimizer es None -> evaluacion (sin gradientes)."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, 100.0 * correct / total


def plot_curves(h, path):
    ep = range(1, len(h["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(ep, h["train_loss"], label="Train", marker="o", ms=3)
    ax[0].plot(ep, h["val_loss"], label="Validacion", marker="o", ms=3)
    ax[0].set_title("Loss (CrossEntropy)"); ax[0].set_xlabel("Epoca"); ax[0].set_ylabel("Loss")
    ax[0].grid(alpha=.3); ax[0].legend()
    ax[1].plot(ep, h["train_acc"], label="Train", marker="o", ms=3)
    ax[1].plot(ep, h["val_acc"], label="Validacion", marker="o", ms=3)
    ax[1].set_title("Accuracy (%)"); ax[1].set_xlabel("Epoca"); ax[1].set_ylabel("Accuracy (%)")
    ax[1].grid(alpha=.3); ax[1].legend()
    fig.suptitle(f"Curvas de aprendizaje - semilla {SEED}")
    plt.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_confusion(cm, path, acc):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right"); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Prediccion"); ax.set_ylabel("Clase real")
    ax.set_title(f"Matriz de confusion (test) - accuracy {acc:.2f}% - semilla {SEED}")
    thr = cm.max() / 2
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thr else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def build_analysis(h, test_acc, test_loss, per_class_acc, cm, best_epoch, n_params):
    """Genera el analisis critico (markdown) con numeros reales de la corrida."""
    tr_acc, va_acc = h["train_acc"][-1], h["val_acc"][-1]
    tr_loss, va_loss = h["train_loss"][-1], h["val_loss"][-1]
    gap_acc = tr_acc - va_acc
    gap_loss = va_loss - tr_loss
    min_val_epoch = int(np.argmin(h["val_loss"])) + 1
    rising = va_loss - min(h["val_loss"])

    if gap_acc > 8 or (gap_acc > 5 and rising > 0.1):
        diag = ("**Sobreajuste (overfitting) evidente**: la brecha train-validacion es grande, "
                "el modelo memoriza el conjunto de entrenamiento.")
    elif tr_acc < 70 and gap_acc < 5:
        diag = ("**Subajuste (underfitting)**: el accuracy es bajo incluso en entrenamiento; "
                "el modelo no tiene capacidad o epocas suficientes.")
    elif gap_acc > 3 or rising > 0.05:
        diag = ("**Sobreajuste leve**: existe una brecha moderada, pero la validacion no se "
                "degrada de forma severa; el modelo generaliza razonablemente bien.")
    else:
        diag = ("**Ajuste adecuado (sin overfitting ni underfitting significativos)**: train y "
                "validacion evolucionan juntas y con brecha pequena.")

    order = np.argsort(per_class_acc)
    worst = ", ".join(f"{CLASSES[i]} ({per_class_acc[i]:.1f}%)" for i in order[:3])
    best = ", ".join(f"{CLASSES[i]} ({per_class_acc[i]:.1f}%)" for i in order[::-1][:3])
    cm_off = cm.copy(); np.fill_diagonal(cm_off, 0)
    i, j = np.unravel_index(cm_off.argmax(), cm_off.shape)

    return f"""### Resultados obtenidos (semilla {SEED})

| Metrica | Entrenamiento | Validacion | Prueba (test) |
|---|---|---|---|
| Accuracy final (epoca {len(h['train_acc'])}) | {tr_acc:.2f}% | {va_acc:.2f}% | {test_acc:.2f}% |
| Loss final | {tr_loss:.4f} | {va_loss:.4f} | {test_loss:.4f} |

- Parametros entrenables: **{n_params:,}**
- Mejor epoca (mayor accuracy de validacion): **{best_epoch}** ({max(h['val_acc']):.2f}%)
- Epoca con menor loss de validacion: **{min_val_epoch}** ({min(h['val_loss']):.4f})
- Brecha de accuracy train - val: **{gap_acc:.2f} puntos porcentuales**
- Brecha de loss val - train: **{gap_loss:.4f}**

![Curvas de aprendizaje](outputs/loss_accuracy.png)

![Matriz de confusion](outputs/confusion_matrix.png)

### Analisis critico

**Diagnostico:** {diag}

- El modelo alcanza **{test_acc:.2f}%** de accuracy en el conjunto de prueba independiente, frente a {tr_acc:.2f}% en entrenamiento y {va_acc:.2f}% en validacion. La diferencia entre validacion y prueba es de {abs(va_acc - test_acc):.2f} puntos, lo que indica que la validacion fue una estimacion fiable de la generalizacion.
- La brecha train-validacion de {gap_acc:.2f} p.p. (y {gap_loss:.4f} en loss) es el indicador cuantitativo principal de overfitting. El loss de validacion alcanzo su minimo en la epoca {min_val_epoch} y al final esta {rising:.4f} por encima de ese minimo.
- Las tecnicas de regularizacion usadas (Data Augmentation con RandomCrop + RandomHorizontalFlip, Dropout 0.5, BatchNorm y weight decay) reducen la memorizacion. Por ello el accuracy de entrenamiento (medido con augmentation activa) puede ser incluso menor o similar al de validacion en las primeras epocas.
- Clases mejor clasificadas: {best}. Clases mas dificiles: {worst}.
- Confusion mas frecuente: **{CLASSES[i]} predicho como {CLASSES[j]}** ({cm_off[i, j]} imagenes), esperable por similitud visual entre ambas clases a 32x32 pixeles.
"""


def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Dispositivo: {DEVICE} | Semilla: {SEED}")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

    model = CNN().to(DEVICE)
    n_params = count_parameters(model)
    print(f"Modelo CNN con {n_params:,} parametros entrenables")

    # 1. Configuracion: CrossEntropyLoss + Adam
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    h = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val, best_epoch = 0.0, 0
    best_path = OUTPUT_DIR / "best_model.pt"

    # 2. Bucle de entrenamiento por epocas
    t0 = time.time()
    for epoch in range(EPOCHS):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion)
        for k, v in zip(h, (tr_loss, tr_acc, va_loss, va_acc)):
            h[k].append(v)
        if va_acc > best_val:
            best_val, best_epoch = va_acc, epoch + 1
            torch.save(model.state_dict(), best_path)
        print(f"Epoca {epoch + 1:02d}/{EPOCHS} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:6.2f}% | "
              f"val loss {va_loss:.4f} acc {va_acc:6.2f}%")
    print(f"Entrenamiento terminado en {(time.time() - t0) / 60:.1f} min")

    # Evaluacion final en TEST con los mejores pesos
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    test_loss, test_acc = run_epoch(model, test_loader, criterion)
    print(f"\nTEST (mejores pesos, epoca {best_epoch}): loss {test_loss:.4f} | accuracy {test_acc:.2f}%")

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds.extend(model(x.to(DEVICE)).argmax(1).cpu().tolist())
            labels.extend(y.tolist())
    cm = confusion_matrix(labels, preds)
    per_class_acc = 100.0 * cm.diagonal() / cm.sum(axis=1)
    print("\n" + classification_report(labels, preds, target_names=CLASSES, digits=3))

    # 3. Artefactos graficos
    plot_curves(h, OUTPUT_DIR / "loss_accuracy.png")
    plot_confusion(cm, OUTPUT_DIR / "confusion_matrix.png", test_acc)

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "epochs": EPOCHS, "history": h, "best_epoch": best_epoch,
                   "test_loss": test_loss, "test_acc": test_acc, "params": n_params,
                   "per_class_acc": dict(zip(CLASSES, per_class_acc.round(2).tolist()))},
                  f, indent=2)

    # 4. Analisis critico -> se inserta solo en el README
    analysis = build_analysis(h, test_acc, test_loss, per_class_acc, cm, best_epoch, n_params)
    readme = ROOT_DIR / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        pattern = r"(<!-- RESULTADOS_START -->).*?(<!-- RESULTADOS_END -->)"
        new = re.sub(pattern, lambda m: f"{m.group(1)}\n{analysis}\n{m.group(2)}", text, flags=re.S)
        readme.write_text(new, encoding="utf-8")
        print("README.md actualizado con los resultados reales.")
    print("Artefactos guardados en outputs/")


if __name__ == "__main__":
    main()
