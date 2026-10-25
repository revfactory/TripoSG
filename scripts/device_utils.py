import torch

DTYPES = {"float16": torch.float16, "float32": torch.float32}


def add_device_args(parser):
    parser.add_argument("--device", type=str, default="auto",
                        help="torch device such as cuda, cuda:1, mps or cpu; auto picks cuda, then mps, then cpu")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", *DTYPES],
                        help="auto uses float16 on cuda and float32 elsewhere")


def resolve_device(device="auto", dtype="auto"):
    """Resolve the device and dtype flags of the command line tools.

    With CUDA available, auto keeps the original choice (cuda, float16).
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device_type = torch.device(device).type  # fail on an invalid device string before downloading weights
    if dtype == "auto":
        return device, torch.float16 if device_type == "cuda" else torch.float32
    return device, DTYPES[dtype]
