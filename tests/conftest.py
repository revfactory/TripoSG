"""TripoSG 테스트 공용 fixture.

사전 학습 가중치 없이 무작위 초기화한 소형 모듈로 파이프라인을 만든다. CUDA와 diso가 없는
환경에서도 돌아가도록 장치를 고를 수 있게 했다. 이 파일은 여러 변경에서 같은 내용으로 추가될 수 있으므로
내용을 바꾸지 않는다. 테스트별 fixture는 각 테스트 파일에 둔다.
"""
import importlib
import sys
import types

import pytest


def _diso_available():
    try:
        importlib.import_module("diso")
        return True
    except Exception:
        return False


DISO_AVAILABLE = _diso_available()


def install_diso_stub():
    """diso가 없을 때 호출하면 예외를 내는 대체 모듈을 넣는다. 이미 있으면 아무것도 하지 않는다."""
    if "diso" in sys.modules or DISO_AVAILABLE:
        return False
    mod = types.ModuleType("diso")

    class DiffDMC:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("diso가 설치되지 않은 테스트 환경이다")

    mod.DiffDMC = DiffDMC
    sys.modules["diso"] = mod
    return True


def available_devices():
    import torch

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def build_tiny_pipeline(seed=0):
    import torch
    from transformers import BitImageProcessor, Dinov2Config, Dinov2Model

    from triposg.models.autoencoders import TripoSGVAEModel
    from triposg.models.transformers import TripoSGDiTModel
    from triposg.pipelines.pipeline_triposg import TripoSGPipeline
    from triposg.schedulers import RectifiedFlowScheduler

    torch.manual_seed(seed)
    return TripoSGPipeline(
        vae=TripoSGVAEModel(
            latent_channels=8, num_attention_heads=2, width_encoder=32, width_decoder=64,
            num_layers_encoder=1, num_layers_decoder=2, embed_frequency=4,
        ),
        transformer=TripoSGDiTModel(
            num_attention_heads=2, width=64, in_channels=8, num_layers=3, cross_attention_dim=32
        ),
        scheduler=RectifiedFlowScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False),
        image_encoder_dinov2=Dinov2Model(
            Dinov2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=64, image_size=56, patch_size=14)
        ),
        feature_extractor_dinov2=BitImageProcessor(
            size={"shortest_edge": 64}, crop_size={"height": 56, "width": 56}, do_center_crop=True,
            image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], resample=3,
        ),
    )


def make_test_image(size=128):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), (255, 255, 255))
    ImageDraw.Draw(img).ellipse((size // 5, size // 6, size * 4 // 5, size * 5 // 6), fill=(200, 80, 40))
    return img


def sphere_logits(radius=0.5, scale=50.0):
    """안쪽이 양수인 해석적 구 로짓 함수. 표면 추출 함수의 geometric_func로 넘긴다."""

    def fn(points):
        return (scale * (radius - points.float().norm(dim=-1, keepdim=True))).to(points.dtype)

    return fn


@pytest.fixture
def tiny_pipe():
    """장치와 dtype을 받아 소형 파이프라인을 만드는 팩토리."""
    install_diso_stub()

    def factory(device="cpu", dtype=None, seed=0):
        import torch

        pipe = build_tiny_pipeline(seed).to(device, dtype or torch.float32)
        for module in (pipe.vae, pipe.transformer, pipe.image_encoder_dinov2):
            module.eval()  # from_pretrained와 같은 추론 모드
        return pipe

    return factory


@pytest.fixture
def cpu_generator():
    import torch

    return lambda seed=0: torch.Generator(device="cpu").manual_seed(seed)
