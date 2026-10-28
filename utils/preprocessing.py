import torch
from torchvision import transforms


IMG_SIZE = 224

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]


# ---------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------

transform = transforms.Compose([

    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


# ---------------------------------------------------------
# Convert normalized tensor back to display image
# ---------------------------------------------------------

def denormalize(tensor):

    image = tensor.cpu().clone()

    for channel, mean, std in zip(
        image,
        IMAGENET_MEAN,
        IMAGENET_STD
    ):

        channel.mul_(std).add_(mean)

    image = torch.clamp(
        image,
        0,
        1
    )

    image = image.numpy()

    # C,H,W → H,W,C
    image = image.transpose(
        1,
        2,
        0
    )

    image = (
        image * 255
    ).astype("uint8")

    return image