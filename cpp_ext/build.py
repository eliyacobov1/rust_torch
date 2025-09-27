import torch
from torch.utils.cpp_extension import load
import pathlib

this_dir = pathlib.Path(__file__).resolve().parent
src = str(this_dir / "backend.cpp")
print("Building PrivateUse1 C++ extension from", src)
ext = load(name="rustcpu_backend", sources=[src], verbose=True)
print("Built:", ext)