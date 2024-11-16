from pathlib import Path
import numpy as np
from PIL import Image

output_path = Path("crafter/minecraft")
output_path.mkdir(exist_ok=True, parents=True)
minecraft_paths = list(Path("crafter/assets_minecraft").iterdir())
crafter_paths = list(Path("crafter/assets").iterdir())

minecraft_names = set([p.name for p in minecraft_paths])
crafter_names = set([p.name for p in crafter_paths])

assert minecraft_names == crafter_names

target_size = (300, 300)

for path in minecraft_paths:
    image = Image.open(path)
    image = image.resize(target_size)
    image.save(output_path / path.name)


# cp crafter/minecraft/coal.png crafter/assets_affine/coal.png
# cp crafter/minecraft/diamond.png crafter/assets_affine/diamond.png
# cp crafter/minecraft/grass.png crafter/assets_affine/grass.png
# cp crafter/minecraft/iron.png crafter/assets_affine/iron.png
# cp crafter/minecraft/lava.png crafter/assets_affine/lava.png
# cp crafter/minecraft/water.png crafter/assets_affine/water.png
# cp crafter/minecraft/wood.png crafter/assets_affine/wood.png