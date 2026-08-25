import torch
from pathlib import Path

def load_linear_from_folder(folder):
    folder = Path(folder)

    W = torch.load(folder / "W_g.pt", map_location="cpu")
    b = torch.load(folder / "b_g.pt", map_location="cpu")

    assert W.ndim == 2
    assert b.ndim == 1
    assert W.shape[0] == b.shape[0]

    model = torch.nn.Linear(W.shape[1], W.shape[0], bias=True)
    model.weight.data.copy_(W)
    model.bias.data.copy_(b)

    return model

modelA = load_linear_from_folder("/leonardo_scratch/fast/IscrC_ARGO/results/CBMs/argo_shapes3d_2026_01_13_22_29_SEED=42_POOLSIZE=462")
#modelA = load_linear_from_folder("/leonardo_scratch/fast/IscrC_ARGO/results/CBMs/argo_shapes3d_2026_01_13_21_01_SEED=42_POOLSIZE=462")
modelB = load_linear_from_folder("/leonardo_scratch/fast/IscrC_ARGO/results/fixed_models/CBM/cbmlite/cbmlite_shapes3d_2026_01_12_15_54_SEED=5_SUBSETSIZE=420")



wA, bA = modelA.weight.cpu(), modelA.bias.cpu()
wB, bB = modelB.weight.cpu(), modelB.bias.cpu()

print("Weights A:\n", wA)
print("Weights B:\n", wB)

print("Bias A:\n", bA)
print("Bias B:\n", bB)

print("Weight L2 diff:", torch.norm(wA - wB).item())
print("Bias L2 diff:", torch.norm(bA - bB).item())