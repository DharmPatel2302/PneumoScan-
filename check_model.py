import torch

path = "model/final_model.pt"

checkpoint = torch.load(
    path,
    map_location="cpu",
    weights_only=False
)

print("\n==============================")
print("MODEL TYPE")
print("==============================")

print(type(checkpoint))


if isinstance(checkpoint, dict):

    print("\n==============================")
    print("CHECKPOINT KEYS")
    print("==============================")

    print(checkpoint.keys())