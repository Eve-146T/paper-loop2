"""Export a trained checkpoint to a self-contained float32 blob for the pure-Kotlin
on-device forward pass (NO onnxruntime — F-Droid clean).

Format "PL21":
    bytes[4]   magic "PL21"
    int32      num tensors
    repeat:
        int32  name length (utf-8)
        bytes  name
        int32  ndim
        int32[ndim] shape
        float32[prod(shape)]  raw row-major little-endian weights

The Kotlin BotNet loads tensors by name and runs the ResNet forward. Run:
    uv run python export.py                       # checkpoints/base/best.pt -> app assets
    uv run python export.py --ckpt checkpoints/league/best.pt
    uv run python export.py --random              # fresh-init net (for the speed benchmark)
"""
import argparse, os, struct
import numpy as np
import torch

from paper_loop.model import ActorCritic, count_params
from paper_loop.rules import OBS_C, OBS_S, OBS_R_DEFAULT

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.normpath(os.path.join(HERE, "..", "app", "src", "main", "assets"))
DEFAULT_OUT = os.path.join(ASSETS, "bot.plb")


class _Head(torch.nn.Module):
    """logits-only head for ONNX export (the on-device bot argmaxes the policy)."""
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, grid, scal):
        logits, _ = self.m(grid, scal)
        return logits


def write_onnx(model, out_path):
    """Single self-contained bot.onnx (weights re-embedded, dynamic batch). Mirrors PL1 export_small."""
    import onnx
    model.eval()
    H = 2 * OBS_R_DEFAULT + 1
    grid = torch.zeros(1, OBS_C, H, H)
    scal = torch.zeros(1, OBS_S)
    torch.onnx.export(
        _Head(model).eval(), (grid, scal), out_path,
        input_names=["grid", "scalar"], output_names=["logits"],
        dynamic_axes={"grid": {0: "n"}, "scalar": {0: "n"}, "logits": {0: "n"}},
        opset_version=17)
    # torch may externalise weights into a sidecar; re-embed so it's ONE file (Gdx opens only bot.onnx)
    m = onnx.load(out_path)
    onnx.save_model(m, out_path, save_as_external_data=False)
    side = out_path + ".data"
    if os.path.exists(side):
        os.remove(side)
    print(f"wrote {out_path}  ({os.path.getsize(out_path)/1024:.1f} KB)")


def write_plb(state_dict, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    items = [(k, v.detach().cpu().float().contiguous().numpy()) for k, v in state_dict.items()]
    with open(out_path, "wb") as f:
        f.write(b"PL21")
        f.write(struct.pack("<i", len(items)))
        for name, t in items:
            nm = name.encode("utf-8")
            f.write(struct.pack("<i", len(nm))); f.write(nm)
            f.write(struct.pack("<i", t.ndim))
            for d in t.shape:
                f.write(struct.pack("<i", int(d)))
            f.write(t.astype("<f4").tobytes())
    sz = os.path.getsize(out_path)
    print(f"wrote {out_path}  ({sz/1024:.1f} KB, {len(items)} tensors)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "base", "best.pt"))
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--scalar_dim", type=int, default=64)
    ap.add_argument("--trunk_dim", type=int, default=128)
    ap.add_argument("--random", action="store_true", help="export a fresh-init net (benchmark only)")
    ap.add_argument("--no_onnx", action="store_true", help="skip bot.onnx (write only the .plb)")
    args = ap.parse_args()

    m = ActorCritic(width=args.width, blocks=args.blocks,
                    scalar_dim=args.scalar_dim, trunk_dim=args.trunk_dim)
    if args.random:
        print(f"exporting RANDOM-init net ({count_params(m)/1e6:.2f}M params) for benchmarking")
    else:
        ck = torch.load(args.ckpt, map_location="cpu")
        sd = ck["model"] if "model" in ck else ck
        m.load_state_dict(sd)
        print(f"loaded {args.ckpt} ({count_params(m)/1e6:.2f}M params)")
    write_plb(m.state_dict(), args.out)                        # pure-Kotlin fallback
    if not args.no_onnx:
        write_onnx(m, os.path.join(os.path.dirname(args.out), "bot.onnx"))   # primary backend


if __name__ == "__main__":
    main()
