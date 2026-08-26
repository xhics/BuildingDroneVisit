from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FloatingOrigin:
    origin_world: np.ndarray
    def __post_init__(self): object.__setattr__(self, "origin_world", np.asarray(self.origin_world, dtype=np.float64))
    def to_render(self, world_xyz: np.ndarray) -> np.ndarray: return (np.asarray(world_xyz,np.float64)-self.origin_world).astype(np.float32)
    def to_world(self, render_xyz: np.ndarray) -> np.ndarray: return np.asarray(render_xyz,np.float64)+self.origin_world


__all__ = ["FloatingOrigin"]
