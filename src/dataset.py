"""
Pregunta 1: carga, visualizacion, transformaciones y particion con semilla.

Semilla individual: 82  (dos ultimos digitos del codigo de matricula)
"""
import random
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ----------------------------------------------------------------------------
# 1. CARGA Y SEMILLA
# ----------------------------------------------------------------------------
SEED = 82  # <-- semilla individual (obligatoria)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

CLASSES = ["avion", "automovil", "pajaro", "gato", "ciervo",
           "perro", "rana", "caballo", "barco", "camion"]

# Estadisticas por canal (R, G, B) de CIFAR-10 (conjunto de entrenamiento)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed: int = SEED) -> None:
    """Fija TODAS las fuentes de aleatoriedad para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ----------------------------------------------------------------------------
# 3. PIPELINE DE TRANSFORMACIONES (con Data Augmentation en entrenamiento)
# ----------------------------------------------------------------------------
def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),         # aumento de datos
        transforms.RandomHorizontalFlip(p=0.5),       # aumento de datos
        transforms.ToTensor(),                        # PIL -> tensor (C,H,W) en [0,1]
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),  # normalizacion de canales RGB
    ])
    # Validacion / prueba: SIN aumento, solo tensor + normalizacion
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_tf, eval_tf


# ----------------------------------------------------------------------------
# 4. PARTICION ESTRATIFICADA (train / val) con la semilla
# ----------------------------------------------------------------------------
def stratified_split(targets, val_ratio: float = 0.2, seed: int = SEED):
    """Devuelve (train_idx, val_idx) manteniendo la proporcion de cada clase."""
    rng = np.random.RandomState(seed)
    targets = np.asarray(targets)
    train_idx, val_idx = [], []
    for c in np.unique(targets):
        idx_c = np.where(targets == c)[0]
        rng.shuffle(idx_c)                            # mezcla con la semilla
        n_val = int(round(len(idx_c) * val_ratio))
        val_idx.extend(idx_c[:n_val].tolist())
        train_idx.extend(idx_c[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def get_dataloaders(batch_size: int = 128, val_ratio: float = 0.2,
                    num_workers: int = 0, verbose: bool = True):
    """Descarga CIFAR-10, aplica transformaciones y devuelve los 3 loaders."""
    set_seed(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_tf, eval_tf = get_transforms()

    # Dos copias del train: una con augmentation y otra sin (para validacion)
    train_full_aug = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_tf)
    train_full_eval = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=eval_tf)
    test_set = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=eval_tf)

    train_idx, val_idx = stratified_split(train_full_aug.targets, val_ratio, SEED)
    train_set = Subset(train_full_aug, train_idx)   # con augmentation
    val_set = Subset(train_full_eval, val_idx)      # sin augmentation

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, worker_init_fn=_seed_worker, generator=g)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if verbose:
        targets = np.array(train_full_aug.targets)
        print(f"[Semilla] {SEED}")
        print(f"[Particion] Entrenamiento : {len(train_set)} imagenes -> tensor (N, 3, 32, 32)")
        print(f"[Particion] Validacion    : {len(val_set)} imagenes -> tensor (N, 3, 32, 32)")
        print(f"[Particion] Prueba (test) : {len(test_set)} imagenes -> tensor (N, 3, 32, 32)")
        print(f"[Estratificacion] imagenes por clase en train: {np.bincount(targets[train_idx], minlength=10).tolist()}")
        print(f"[Estratificacion] imagenes por clase en val  : {np.bincount(targets[val_idx], minlength=10).tolist()}")
        xb, yb = next(iter(train_loader))
        print(f"[Batch] x: {tuple(xb.shape)} (Batch, Channels, Height, Width) | y: {tuple(yb.shape)}")

    return train_loader, val_loader, test_loader


# ----------------------------------------------------------------------------
# 2. VISUALIZACION DEL DATASET (cuadricula con etiquetas reales)
# ----------------------------------------------------------------------------
def denormalize(img: torch.Tensor) -> np.ndarray:
    """Tensor normalizado (C,H,W) -> array (H,W,C) en [0,1] para graficar."""
    mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(3, 1, 1)
    return (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def _finish(fig, save_path, show):
    import matplotlib.pyplot as plt
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show and "agg" not in matplotlib.get_backend().lower():
        plt.show()
    plt.close(fig)


def show_samples(n: int = 16, save_path=None, show: bool = True):
    """Grafica una cuadricula de n imagenes (8 a 16) con su clase real."""
    import matplotlib.pyplot as plt
    set_seed(SEED)
    raw = datasets.CIFAR10(DATA_DIR, train=True, download=True)  # imagenes PIL originales
    idx = np.random.RandomState(SEED).choice(len(raw), n, replace=False)

    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.8 * rows))
    for ax, i in zip(np.ravel(axes), idx):
        img, label = raw[i]
        ax.imshow(img)
        ax.set_title(f"{CLASSES[label]}", fontsize=11)
        ax.axis("off")
    fig.suptitle(f"CIFAR-10 - {n} muestras aleatorias (semilla {SEED})", fontsize=14)
    _finish(fig, save_path, show)


def show_augmentation(save_path=None, show: bool = True):
    """Muestra una misma imagen con distintas transformaciones aleatorias."""
    import matplotlib.pyplot as plt
    set_seed(SEED)
    raw = datasets.CIFAR10(DATA_DIR, train=True, download=True)
    train_tf, _ = get_transforms()
    img, label = raw[0]
    fig, axes = plt.subplots(1, 6, figsize=(14, 2.8))
    axes[0].imshow(img)
    axes[0].set_title(f"Original\n({CLASSES[label]})")
    axes[0].axis("off")
    for ax in axes[1:]:
        ax.imshow(denormalize(train_tf(img)))
        ax.set_title("Augmentada")
        ax.axis("off")
    _finish(fig, save_path, show)


if __name__ == "__main__":
    get_dataloaders()
    show_samples(16, save_path=OUTPUT_DIR / "sample_grid.png")
    show_augmentation(save_path=OUTPUT_DIR / "augmentation_examples.png")
    print("Listo: imagenes guardadas en outputs/")
