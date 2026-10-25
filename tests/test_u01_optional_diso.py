"""U01: diso is an optional dependency.

Without diso or a CUDA device the pipelines must still import and the default call
(use_flash_decoder=True) must fall back to hierarchical extraction. With diso and CUDA the
flash path must be used with the same arguments as before.
"""
import logging
import os
import subprocess
import sys
import types

import numpy as np
import pytest
import torch

from conftest import available_devices, install_diso_stub, make_test_image, sphere_logits

BOUND = 1.005
DEFAULT_BOUNDS = (-1.005, -1.005, -1.005, 1.005, 1.005, 1.005)


def _repo_root():
    # The triposg found on sys.path (the current directory under `python -m pytest`).
    import triposg

    return os.path.dirname(list(triposg.__path__)[0])


def test_import_without_diso():
    code = "\n".join([
        "import sys",
        "sys.modules['diso'] = None",  # `import diso` now raises ImportError
        "import triposg.pipelines.pipeline_triposg",
        "import triposg.pipelines.pipeline_triposg_scribble",
        "import triposg.inference_utils as iu",
        "assert iu.DiffDMC is None, iu.DiffDMC",
    ])
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=_repo_root(), capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, proc.stderr[-3000:]


@pytest.mark.parametrize("device", available_devices())
def test_hierarchical_refinement_runs_without_cuda(device):
    install_diso_stub()
    from triposg.inference_utils import hierarchical_extract_geometry

    ((vertices, faces),) = hierarchical_extract_geometry(
        sphere_logits(), torch.device(device), bounds=BOUND,
        dense_octree_depth=5, hierarchical_octree_depth=6,
    )
    assert vertices is not None and len(faces) > 0
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    assert np.abs(center).max() < 2 * BOUND / 2**6


def test_default_call_falls_back_without_flash(tiny_pipe, cpu_generator, caplog):
    pipe = tiny_pipe("cpu")
    kwargs = dict(image=make_test_image(), num_inference_steps=2, num_tokens=32,
                  dense_octree_depth=5, hierarchical_octree_depth=5)

    with caplog.at_level(logging.WARNING):
        out = pipe(generator=cpu_generator(), **kwargs)  # use_flash_decoder defaults to True
    assert len(out.samples) == 1 and len(out.meshes) == 1
    assert any(r.levelno == logging.WARNING and "diso" in r.getMessage() for r in caplog.records)

    ref = pipe(generator=cpu_generator(), use_flash_decoder=False, **kwargs)
    np.testing.assert_array_equal(out.samples[0][0], ref.samples[0][0])
    np.testing.assert_array_equal(out.samples[0][1], ref.samples[0][1])


def test_flash_extract_geometry_requires_diso(monkeypatch):
    install_diso_stub()
    import triposg.inference_utils as iu

    decode_calls = []

    class FakeVAE:
        decoder = types.SimpleNamespace(set_topk=lambda topk: None)

        def decode(self, z, sampled_points, **kwargs):
            decode_calls.append(tuple(sampled_points.shape))
            return types.SimpleNamespace(sample=sphere_logits()(sampled_points))

    monkeypatch.setattr(iu, "DiffDMC", None)
    with pytest.raises(ImportError, match="diso"):
        iu.flash_extract_geometry(torch.zeros(1, 4, 4), FakeVAE(), bounds=BOUND, octree_depth=6)
    assert decode_calls == []


def test_flash_extraction_available(monkeypatch):
    install_diso_stub()
    import triposg.inference_utils as iu

    monkeypatch.setattr(iu, "DiffDMC", object)  # stand-in for diso.DiffDMC
    assert iu.flash_extraction_available(torch.device("cuda")) is True
    assert iu.flash_extraction_available(torch.device("cuda", 1)) is True
    assert iu.flash_extraction_available(torch.device("cpu")) is False
    assert iu.flash_extraction_available(torch.device("mps")) is False

    monkeypatch.setattr(iu, "DiffDMC", None)
    assert iu.flash_extraction_available(torch.device("cuda")) is False


def test_flash_path_used_when_available(tiny_pipe, cpu_generator, monkeypatch):
    import triposg.pipelines.pipeline_triposg as pt

    pipe = tiny_pipe("cpu")
    flash_calls, set_flash_calls = [], []

    def fake_flash(latents, vae, **kwargs):
        flash_calls.append({"shape": tuple(latents.shape), "vae": vae, **kwargs})
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        return [(vertices, np.array([[0, 1, 2]]))]

    def no_hierarchical(*args, **kwargs):
        raise AssertionError("hierarchical extraction must not run when the flash path is available")

    monkeypatch.setattr(pt, "flash_extraction_available", lambda device: True)
    monkeypatch.setattr(pt, "flash_extract_geometry", fake_flash)
    monkeypatch.setattr(pt, "hierarchical_extract_geometry", no_hierarchical)
    monkeypatch.setattr(pipe.vae, "set_flash_decoder", lambda: set_flash_calls.append(True))

    out = pipe(image=make_test_image(), num_inference_steps=2, num_tokens=32,
               generator=cpu_generator(), flash_octree_depth=7)

    assert len(set_flash_calls) == 1
    assert len(flash_calls) == 1
    call = flash_calls[0]
    assert call["shape"] == (1, 32, 8)
    assert call["vae"] is pipe.vae
    assert tuple(call["bounds"]) == DEFAULT_BOUNDS
    assert call["octree_depth"] == 7
    assert len(out.meshes) == 1
