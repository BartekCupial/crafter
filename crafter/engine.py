import collections
import functools
import pathlib
import copy

import imageio.v3 as imageio
import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path
import open3d as o3d
import tempfile

class AttrDict(dict):

  __getattr__ = dict.__getitem__


class staticproperty:

  def __init__(self, function):
    self.function = function

  def __get__(self, instance, owner=None):
    return self.function()


class World:

  def __init__(self, area, materials, chunk_size):
    self.area = area
    self._chunk_size = chunk_size
    self._mat_names = {i: x for i, x in enumerate([None] + materials)}
    self._mat_ids = {x: i for i, x in enumerate([None] + materials)}
    self.reset()

  def reset(self, seed=None):
    self.random = np.random.RandomState(seed)
    self.daylight = 0.0
    self._chunks = collections.defaultdict(set)
    self._objects = [None]
    self._mat_map = np.zeros(self.area, np.uint8)
    self._obj_map = np.zeros(self.area, np.uint32)

  @property
  def objects(self):
    # Return a new list so the objects cannot change while being iterated over.
    return [obj for obj in self._objects if obj]

  @property
  def chunks(self):
    return self._chunks.copy()

  def add(self, obj):
    assert hasattr(obj, 'pos')
    obj.pos = np.array(obj.pos)
    assert self._obj_map[tuple(obj.pos)] == 0
    index = len(self._objects)
    self._objects.append(obj)
    self._obj_map[tuple(obj.pos)] = index
    self._chunks[self.chunk_key(obj.pos)].add(obj)

  def remove(self, obj):
    if obj.removed:
      return
    self._objects[self._obj_map[tuple(obj.pos)]] = None
    self._obj_map[tuple(obj.pos)] = 0
    self._chunks[self.chunk_key(obj.pos)].remove(obj)
    obj.removed = True

  def move(self, obj, pos):
    if obj.removed:
      return
    pos = np.array(pos)
    assert self._obj_map[tuple(pos)] == 0
    index = self._obj_map[tuple(obj.pos)]
    self._obj_map[tuple(pos)] = index
    self._obj_map[tuple(obj.pos)] = 0
    old_chunk = self.chunk_key(obj.pos)
    new_chunk = self.chunk_key(pos)
    if old_chunk != new_chunk:
      self._chunks[old_chunk].remove(obj)
      self._chunks[new_chunk].add(obj)
    obj.pos = pos

  def __setitem__(self, pos, material):
    if material not in self._mat_ids:
      id_ = len(self._mat_ids)
      self._mat_ids[material] = id_
    self._mat_map[tuple(pos)] = self._mat_ids[material]

  def __getitem__(self, pos):
    if not _inside((0, 0), pos, self.area):
      return None, None
    material = self._mat_names[self._mat_map[tuple(pos)]]
    obj = self._objects[self._obj_map[tuple(pos)]]
    return material, obj

  def nearby(self, pos, distance):
    (x, y), d = pos, distance
    ids = set(self._mat_map[
        x - d: x + d + 1, y - d: y + d + 1].flatten().tolist())
    materials = tuple(self._mat_names[x] for x in ids)
    indices = self._obj_map[
        x - d: x + d + 1, y - d: y + d + 1].flatten().tolist()
    objs = {self._objects[i] for i in indices if i > 0}
    return materials, objs

  def mask(self, xmin, xmax, ymin, ymax, material):
    region = self._mat_map[xmin: xmax, ymin: ymax]
    return (region == self._mat_ids[material])

  def count(self, material):
    return (self._mat_map == self._mat_ids[material]).sum()

  def chunk_key(self, pos):
    (x, y), (csx, csy) = pos, self._chunk_size
    xmin, ymin = (x // csx) * csx, (y // csy) * csy
    xmax = min(xmin + csx, self.area[0])
    ymax = min(ymin + csy, self.area[1])
    return (xmin, xmax, ymin, ymax)


class Textures:

  def __init__(self, directory):
    self._originals = {}
    self._textures = {}
    for filename in pathlib.Path(directory).glob('*.png'):
      image = imageio.imread(filename.read_bytes())
      image = image.transpose((1, 0) + tuple(range(2, len(image.shape))))
      self._originals[filename.stem] = image
      self._textures[(filename.stem, image.shape[:2])] = image

  def get(self, name, size):
    if name is None:
      name = 'unknown'
    size = int(size[0]), int(size[1])
    key = name, size
    if key not in self._textures:
      image = self._originals[name]
      image = Image.fromarray(image)
      image = image.resize(size[::-1], resample=Image.NEAREST)
      image = np.array(image)
      self._textures[key] = image
    return self._textures[key]


class GlobalView:

  pass


class UncoverView:

  pass


class LocalView:

  def __init__(self, world, textures, grid):
    self._world = world
    self._textures = textures
    self._grid = np.array(grid)
    self._offset = self._grid // 2
    self._area = np.array(self._world.area)
    self._center = None
    
    self.meshes = {}
    for i, mesh_path in enumerate(Path("/home/bartek/Workspace/ideas/crafter/glb").iterdir()):
        mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=True)
        mesh.compute_vertex_normals()
        
        bbox = mesh.get_axis_aligned_bounding_box()
        scale = 1.0 / np.max(bbox.get_extent())
        mesh.scale(scale, center=bbox.get_center())
        mesh.translate(-mesh.get_center())
  
        if mesh_path.stem in ["skeleton", "player", "zombie"]:
          mesh.scale(1.6, center=bbox.get_center())
        elif mesh_path.stem in ["tree"]:
          mesh.scale(1.5, center=bbox.get_center())
          mesh.translate([0, 1.0, 0])
        elif mesh_path.stem in ["table", "furnace"]:
          mesh.scale(0.8, center=bbox.get_center())
        # elif mesh_path.stem in ["cow"]:
        #   mesh.scale(1.2, center=bbox.get_center())
          
        # mesh.translate([1.0 * i, 0, 0])
        self.meshes[mesh_path.stem] = mesh 

  def __call__(self, player, unit):
    self._unit = np.array(unit)
    self._center = np.array(player.pos)
    canvas = np.zeros(tuple(self._grid * unit) + (4,), np.uint8) + 127
    
    top_textures_names = ["tree", "coal", "diamond", "iron", "stone", "wood", "table", "furnace"]

    mesh_world = []
    for x in range(-2, self._grid[0] + 2):
        for y in range(-2, self._grid[1] + 2):
            pos = self._center + np.array([x, y]) - self._offset
            if not _inside((0, 0), pos, self._area):
              continue
            
            material = self._world[pos][0]
            if material in top_textures_names:
              mesh = copy.deepcopy(self.meshes[material])
              mesh.translate([x, 1, y])
              mesh_world.append(mesh)
              material = "grass"
              
            mesh = copy.deepcopy(self.meshes[material])
            mesh.translate([x, 0, y])
            mesh_world.append(mesh)
            
    for obj in self._world.objects:
      pos = obj.pos - self._center + self._offset
      if not _inside((0, 0), pos, self._grid):
        continue

      material = obj.texture
      if "player" in material or "arrow" in material:
        material, dir = material.split("-")
        mesh = copy.deepcopy(self.meshes[material])
        if dir == "down":
          R = mesh.get_rotation_matrix_from_xyz((0, 0, 0))
          mesh.rotate(R, center=(0, 0, 0))
        elif dir == "up":
          R = mesh.get_rotation_matrix_from_xyz((0, np.pi, 0))
          mesh.rotate(R, center=(0, 0, 0))
        elif dir == "left":
          R = mesh.get_rotation_matrix_from_xyz((0, -np.pi / 2.2, 0))
          mesh.rotate(R, center=(0, 0, 0))
        elif dir == "right":
          R = mesh.get_rotation_matrix_from_xyz((0, np.pi / 2.2, 0))
          mesh.rotate(R, center=(0, 0, 0))
      else:
        mesh = copy.deepcopy(self.meshes[material])
      
      x, y = pos
      mesh.translate([x, 1, y])
      mesh_world.append(mesh)
    
    width, height, *_ = canvas.shape
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    
    for mesh in mesh_world:
      vis.add_geometry(mesh)

    # Get the view control
    view_control = vis.get_view_control()

    # Set camera position and orientation
    # Method 1: Set camera parameters directly
    view_control.set_front([0, 1, 0.5])    # Camera looks straight down (-y direction)
    view_control.set_lookat([4, 0, 3])    # Looking at point (4, 0, 3)
    view_control.set_up([0, 1, 0])        # Up direction aligned with x-axis
    view_control.set_zoom(0.3)            # Zoom level

    # Render and capture
    vis.poll_events()
    vis.update_renderer()
    tf = tempfile.NamedTemporaryFile()
    vis.capture_screen_image(tf.name + ".png")

    vis.destroy_window()
    
    image = Image.open(tf.name + ".png")
    canvas = np.asanyarray(image)
    
    canvas = self._light(canvas, self._world.daylight)
    if player.sleeping:
      canvas = self._sleep(canvas)
    if player.health < 1:
      canvas = self._tint(canvas, (128, 0, 0), 0.6)
    return canvas.transpose((1, 0, 2))

  def _light(self, canvas, daylight):
    night = canvas
    if daylight < 0.5:
      night = self._noise(night, 2 * (0.5 - daylight), 0.5)
    night = np.array(ImageEnhance.Color(Image.fromarray(night.astype(np.uint8))).enhance(0.4))
    night = self._tint(night, (0, 16, 64), 0.5)
    return daylight * canvas + (1 - daylight) * night

  def _sleep(self, canvas):
    canvas = np.array(ImageEnhance.Color(
        Image.fromarray(canvas.astype(np.uint8))).enhance(0.0))
    canvas = self._tint(canvas, (0, 0, 16), 0.5)
    return canvas

  def _tint(self, canvas, color, amount):
    color = np.array(color)
    return (1 - amount) * canvas + amount * color

  def _noise(self, canvas, amount, stddev):
    noise = self._world.random.uniform(32, 127, canvas.shape[:2])[..., None]
    mask = amount * self._vignette(canvas.shape, stddev)[..., None]
    return (1 - mask) * canvas + mask * noise

  @functools.lru_cache(10)
  def _vignette(self, shape, stddev):
    xs, ys = np.meshgrid(
        np.linspace(-1, 1, shape[0]),
        np.linspace(-1, 1, shape[1]))
    return 1 - np.exp(-0.5 * (xs ** 2 + ys ** 2) / (stddev ** 2)).T


class ItemView:

  def __init__(self, textures, grid):
    self._textures = textures
    self._grid = np.array(grid)

  def __call__(self, inventory, unit):
    unit = np.array(unit)
    canvas = np.zeros(tuple(self._grid * unit) + (3,), np.uint8)
    for index, (item, amount) in enumerate(inventory.items()):
      if amount < 1:
        continue
      self._item(canvas, index, item, unit)
      self._amount(canvas, index, amount, unit)
    return canvas

  def _item(self, canvas, index, item, unit):
    pos = index % self._grid[0], index // self._grid[0]
    pos = (pos * unit + 0.1 * unit).astype(np.int32)
    texture = self._textures.get(item, 0.8 * unit)
    _draw(canvas, pos, texture)

  def _amount(self, canvas, index, amount, unit):
    pos = index % self._grid[0], index // self._grid[0]
    pos = (pos * unit + 0.4 * unit).astype(np.int32)
    text = str(amount) if amount in list(range(10)) else 'unknown'
    texture = self._textures.get(text, 0.6 * unit)
    _draw(canvas, pos, texture)


class SemanticView:

  def __init__(self, world, obj_types):
    self._world = world
    self._mat_ids = world._mat_ids.copy()
    self._obj_ids = {
        c: len(self._mat_ids) + i
        for i, c in enumerate(obj_types)}

  def __call__(self):
    canvas = self._world._mat_map.copy()
    for obj in self._world.objects:
      canvas[tuple(obj.pos)] = self._obj_ids[type(obj)]
    return canvas


def _inside(lhs, mid, rhs):
  return (lhs[0] <= mid[0] < rhs[0]) and (lhs[1] <= mid[1] < rhs[1])

def _draw(canvas, pos, texture):
  (x, y), (w, h) = pos, texture.shape[:2]

  # Get the target area in the canvas
  target_area = canvas[x:x + w, y:y + h]

  if texture.shape[-1] == 4:
    # If texture has alpha channel
    rgb = texture[..., :3]
    alpha = texture[..., 3:4] / 255.0  # Normalize alpha to 0-1
    
    # New = alpha * foreground + (1 - alpha) * background
    blended_rgb = (alpha * rgb + (1 - alpha) * target_area[..., :3])
    blended_alpha = alpha + (1 - alpha) * (target_area[..., 3:4] / 255.0)

    # Combine RGB and alpha
    canvas[x:x + w, y:y + h] = np.concatenate([blended_rgb, blended_alpha * 255], axis=-1)
  else:
    # If texture has only RGB channels, assume full opacity
    canvas[x:x + w, y:y + h, :3] = texture
    canvas[x:x + w, y:y + h, 3] = 255

def _draw_alpha(canvas, pos, texture):
  (x, y), (w, h) = pos, texture.shape[:2]
  if texture.shape[-1] == 4:
    alpha = texture[..., 3:].astype(np.float32) / 255
    texture = texture[..., :3].astype(np.float32) / 255
    current = canvas[x: x + w, y: y + h].astype(np.float32) / 255
    blended = alpha * texture + (1 - alpha) * current
    texture = (255 * blended).astype(np.uint8)
  canvas[x: x + w, y: y + h] = texture
