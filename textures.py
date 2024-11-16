from pathlib import Path
from PIL import Image
import numpy as np

textures = list(Path("crafter/minecraft").iterdir())

for texture in textures:
    img = Image.open(texture)
    print(f"{texture.name}: {np.asanyarray(img).shape}")
