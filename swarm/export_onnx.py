"""Phase 2 — export the trained MAPPO actor to ONNX and prove parity.

Loads ``swarm/checkpoints/policy.pt``, rebuilds the decentralized ``Actor``,
and exports its deterministic inference graph (``Actor.forward(obs) -> action``)
to ONNX with a dynamic batch axis so any number of agents N can be evaluated in
one call. Then it runs a parity test: identical random + realistic obs through
PyTorch and onnxruntime, asserting the max abs diff is within tolerance, at two
different batch sizes (to confirm the dynamic axis works).

Run:
    PYTHONPATH=/Users/nikhi/bow-capital-hackathon \\
        uv run python -m swarm.export_onnx

Outputs:
    frontend/public/policy.onnx        (loaded by Phase 3 onnxruntime-web)
    swarm/checkpoints/policy.onnx      (local copy)

ONNX inference contract (for Phase 3 / onnxruntime-web):
    input  name "obs"     float32  shape (N, 36)   dynamic axis 0 = "batch"
    output name "action"  float32  shape (N, 2)    deterministic, in [-1, 1]
"""

from __future__ import annotations

import os

import numpy as np
import onnxruntime as ort
import torch

from swarm.models import Actor

# ----------------------------------------------------------------- paths ---
_HERE = os.path.dirname(__file__)
_REPO = os.path.dirname(_HERE)
CKPT_PATH = os.path.join(_HERE, "checkpoints", "policy.pt")
FRONTEND_ONNX = os.path.join(_REPO, "frontend", "public", "policy.onnx")
CKPT_ONNX = os.path.join(_HERE, "checkpoints", "policy.onnx")

OPSET = 17
PARITY_TOL = 1e-5
PARITY_BATCHES = (5, 37)  # exercise the dynamic batch axis


def load_actor() -> tuple[Actor, dict]:
    """Rebuild the trained actor from the checkpoint, in eval mode."""
    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    actor = Actor(ck["obs_dim"], ck["act_dim"], ck["actor_hidden"], ck["log_std_init"])
    actor.load_state_dict(ck["actor_state_dict"])
    actor.eval()
    return actor, ck


def export(actor: Actor, obs_dim: int) -> None:
    """Export Actor.forward to ONNX with a dynamic batch axis (static feat dim)."""
    os.makedirs(os.path.dirname(FRONTEND_ONNX), exist_ok=True)
    os.makedirs(os.path.dirname(CKPT_ONNX), exist_ok=True)

    # Example input: batch of 1 is fine; the batch axis is marked dynamic below.
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)

    torch.onnx.export(
        actor,
        dummy,
        FRONTEND_ONNX,
        dynamo=False,                 # legacy exporter (no onnxscript dependency)
        opset_version=OPSET,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        do_constant_folding=True,
    )

    # keep a local copy alongside the checkpoint
    import shutil

    shutil.copyfile(FRONTEND_ONNX, CKPT_ONNX)


def make_obs(batch: int, obs_dim: int, rng: np.random.Generator) -> np.ndarray:
    """A mix of plausible obs values: positions/velocities in [-1, 1] plus some
    wider noise to stress the net beyond the training distribution."""
    # half the rows in the [-1, 1] realistic range, half wider Gaussian noise
    realistic = rng.uniform(-1.0, 1.0, size=(batch, obs_dim))
    wide = rng.normal(0.0, 1.5, size=(batch, obs_dim))
    mask = rng.random((batch, 1)) < 0.5
    obs = np.where(mask, realistic, wide)
    return obs.astype(np.float32)


def parity_test(actor: Actor) -> bool:
    """Run PyTorch vs onnxruntime over multiple batch sizes; assert parity."""
    sess = ort.InferenceSession(FRONTEND_ONNX, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    rng = np.random.default_rng(0)
    overall_max = 0.0
    all_pass = True

    for b in PARITY_BATCHES:
        obs = make_obs(b, actor.obs_dim, rng)

        with torch.no_grad():
            torch_out = actor(torch.from_numpy(obs)).numpy()

        ort_out = sess.run([out_name], {in_name: obs})[0]

        max_diff = float(np.abs(torch_out - ort_out).max())
        overall_max = max(overall_max, max_diff)
        ok = max_diff < PARITY_TOL
        all_pass = all_pass and ok
        print(f"  batch={b:3d}  torch{torch_out.shape} vs onnx{ort_out.shape}  "
              f"max_abs_diff={max_diff:.3e}  {'PASS' if ok else 'FAIL'}")

    print(f"\nONNX I/O: input='{in_name}' output='{out_name}' "
          f"(dynamic axis 0 = batch, feat dim = {actor.obs_dim} -> "
          f"act dim = {actor.act_dim})")
    print(f"overall max_abs_diff = {overall_max:.3e}  (tol {PARITY_TOL:.0e})")
    print("PARITY: PASS" if all_pass else "PARITY: FAIL")
    return all_pass


def main() -> None:
    actor, ck = load_actor()
    print(f"[load] actor rebuilt from {CKPT_PATH} "
          f"(obs_dim={ck['obs_dim']}, act_dim={ck['act_dim']}, "
          f"hidden={ck['actor_hidden']})")

    export(actor, ck["obs_dim"])
    size = os.path.getsize(FRONTEND_ONNX)
    print(f"[export] wrote {FRONTEND_ONNX} ({size} bytes)")
    print(f"[export] copied  {CKPT_ONNX}")

    ok = parity_test(actor)
    if not ok:
        raise SystemExit("parity test FAILED — export not trustworthy")


if __name__ == "__main__":
    main()
