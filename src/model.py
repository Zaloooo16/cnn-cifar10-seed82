"""
Pregunta 3: arquitectura CNN y validacion del flujo de tensores 4D.

Flujo de shapes (Batch = B):
  Entrada            (B,   3, 32, 32)
  Bloque 1  Conv2d(3->32)   + BN + ReLU   (B,  32, 32, 32)
            MaxPool2d(2)                  (B,  32, 16, 16)
  Bloque 2  Conv2d(32->64)  + BN + ReLU   (B,  64, 16, 16)
            MaxPool2d(2)                  (B,  64,  8,  8)
  Bloque 3  Conv2d(64->128) + BN + ReLU   (B, 128,  8,  8)
            MaxPool2d(2)                  (B, 128,  4,  4)
  Flatten                                 (B, 2048)
  Linear(2048->256) + ReLU + Dropout      (B, 256)
  Linear(256->10)                         (B, 10)
"""
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        # ---- Bloques convolucionales (Conv2d -> ReLU -> MaxPool2d) ----
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # ---- Conexion con las capas densas: aplanado ----
        self.flatten = nn.Flatten()
        # 128 canales * 4 * 4 = 2048 (32 -> 16 -> 8 -> 4 por los 3 MaxPool)
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f"Se esperaba un tensor 4D (B,C,H,W), llego {tuple(x.shape)}"
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        return self.classifier(x)

    @torch.no_grad()
    def trace_shapes(self, x: torch.Tensor) -> torch.Tensor:
        """Pasa x por el modelo imprimiendo el shape tras cada etapa."""
        print(f"{'Entrada':<28}{tuple(x.shape)}")
        for name, layer in [("Bloque 1 (Conv-BN-ReLU-Pool)", self.block1),
                            ("Bloque 2 (Conv-BN-ReLU-Pool)", self.block2),
                            ("Bloque 3 (Conv-BN-ReLU-Pool)", self.block3),
                            ("Flatten", self.flatten),
                            ("Clasificador (Linear)", self.classifier)]:
            x = layer(x)
            print(f"{name:<28}{tuple(x.shape)}")
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # ---------------- Validacion de shapes (aserciones) ----------------
    BATCH = 8
    model = CNN()
    model.eval()
    dummy = torch.randn(BATCH, 3, 32, 32)  # lote simulado (Batch, Channels, Height, Width)

    print("=== Validacion del forward pass con un batch simulado ===")
    out = model.trace_shapes(dummy)

    assert dummy.ndim == 4, "La entrada debe ser 4D"
    assert out.shape == (BATCH, 10), f"Salida inesperada: {tuple(out.shape)}"
    assert torch.equal(out, model(dummy)), "trace_shapes y forward deben coincidir"
    print(f"\nOK: entrada {tuple(dummy.shape)} -> salida {tuple(out.shape)} (10 clases)")
    print(f"Parametros entrenables: {count_parameters(model):,}")
