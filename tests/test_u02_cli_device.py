import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from conftest import DISO_AVAILABLE


def repo_root():
    """Root of the repository under test: the one whose triposg package is imported."""
    spec = importlib.util.find_spec("triposg")
    if spec is None or not spec.submodule_search_locations:
        return Path(__file__).resolve().parents[1]
    return Path(list(spec.submodule_search_locations)[0]).resolve().parent


def test_resolve_device_prefers_cuda(monkeypatch):
    from scripts.device_utils import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device() == ("cuda", torch.float16)
    assert resolve_device("auto", "auto") == ("cuda", torch.float16)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto", "auto") == ("mps", torch.float32)

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto", "auto") == ("cpu", torch.float32)

    # explicit values are kept; auto dtype follows the device type
    assert resolve_device("cuda:1", "auto") == ("cuda:1", torch.float16)
    assert resolve_device("mps", "float16") == ("mps", torch.float16)
    assert resolve_device("cpu", "float32") == ("cpu", torch.float32)


class CpuRmbgStandIn(torch.nn.Module):
    """Stand-in for BriaRMBG on CPU: result[0][0] is a soft disc mask of shape (1, 1, H, W)."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)  # keeps the module on CPU

    def forward(self, image):
        assert image.device.type == "cpu"
        _, _, h, w = image.shape
        ys = torch.linspace(-1, 1, h).view(h, 1)
        xs = torch.linspace(-1, 1, w).view(1, w)
        mask = torch.sigmoid(20 * (0.6 - (xs ** 2 + ys ** 2).sqrt()))
        return [mask.view(1, 1, h, w)], None


def write_valid_alpha_png(path, size=256):
    bgra = np.zeros((size, size, 4), dtype=np.uint8)
    bgra[..., :3] = (40, 80, 200)
    cv2.circle(bgra, (size // 2, size // 2), size // 3, (40, 80, 200, 255), -1)
    cv2.imwrite(str(path), bgra)
    return path


@pytest.mark.parametrize(
    "image",
    ["assets/example_scribble_data/cat_with_wings.png", "assets/example_data/hjswed.png", "valid_alpha"],
    ids=["rgb", "rgba_opaque", "rgba_valid_alpha"],
)
def test_load_image_runs_on_cpu(image, tmp_path):
    from scripts.image_process import load_image, prepare_image

    path = write_valid_alpha_png(tmp_path / "valid_alpha.png") if image == "valid_alpha" else repo_root() / image
    rmbg = CpuRmbgStandIn()
    bg_color = np.array([1.0, 1.0, 1.0])

    out = load_image(str(path), bg_color=bg_color, rmbg_net=rmbg)
    assert isinstance(out, torch.Tensor), out
    assert out.device.type == "cpu"
    assert out.ndim == 3 and out.shape[0] == 3

    img = prepare_image(str(path), bg_color=bg_color, rmbg_net=rmbg, device="cpu")
    assert isinstance(img, Image.Image)


@pytest.mark.parametrize(
    "module", ["scripts.inference_triposg", "scripts.inference_triposg_scribble", "scripts.inference_vae"]
)
def test_cli_help_lists_device_flags(module, tmp_path):
    env = dict(os.environ)
    if not DISO_AVAILABLE:
        # diso is a CUDA extension; a stub package lets the tools import far enough to print --help
        (tmp_path / "diso").mkdir()
        (tmp_path / "diso" / "__init__.py").write_text(
            "class DiffDMC:\n    def __init__(self, *args, **kwargs):\n        raise RuntimeError('diso stub')\n"
        )
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(tmp_path), env.get("PYTHONPATH")) if p)
    proc = subprocess.run(
        [sys.executable, "-m", module, "--help"], cwd=repo_root(), env=env,
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "--device" in proc.stdout
    assert "--dtype" in proc.stdout
