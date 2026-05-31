"""Phase 2 — export the trained MAPPO actor to ONNX and prove parity.

Loads ``swarm/checkpoints/<env_id>/policy.pt``, rebuilds the decentralized
``Actor``, and exports its deterministic inference graph
(``Actor.forward(obs) -> action``) to ONNX with a dynamic batch axis so any
number of agents N can be evaluated in one call.

Run:
    uv run --project swarm python -m swarm.export_onnx --env-id search-and-interdict

Outputs:
    frontend/public/policies/<env_id>/policy.onnx
    swarm/checkpoints/<env_id>/policy.onnx

ONNX inference contract (for Phase 3 / onnxruntime-web):
    input  name "obs"     float32  shape (N, OBS_DIM)   dynamic axis 0 = "batch"
    output name "action"  float32  shape (N, 2)         deterministic, in [-1, 1]

``OBS_DIM`` is recorded in the checkpoint and propagated through the export so
the resulting ONNX is always shape-correct for whatever the env was trained on
(48 with scenario obstacles enabled; 36 in the obstacle-free legacy build).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import onnxruntime as ort
import torch

from swarm.models import Actor

HERE = os.path.dirname(__file__)
REPO = os.path.dirname(HERE)

OPSET = 17
PARITY_TOL = 1e-5
PARITY_BATCHES = (5, 37)


def checkpoint_dir(env_id: str) -> str:
    return os.path.join(HERE, "checkpoints", env_id)


def checkpoint_paths(env_id: str) -> dict[str, str]:
    ckpt_dir = checkpoint_dir(env_id)
    return {
        "checkpoint": os.path.join(ckpt_dir, "policy.pt"),
        "onnx": os.path.join(ckpt_dir, "policy.onnx"),
        "meta": os.path.join(ckpt_dir, "meta.json"),
        "frontend": os.path.join(REPO, "frontend", "public", "policies", env_id, "policy.onnx"),
    }


def load_actor(ckpt_path: str) -> tuple[Actor, dict]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    actor = Actor(ck["obs_dim"], ck["act_dim"], ck["actor_hidden"], ck["log_std_init"])
    actor.load_state_dict(ck["actor_state_dict"])
    actor.eval()
    return actor, ck


def export(actor: Actor, obs_dim: int, frontend_path: str, checkpoint_path: str) -> None:
    os.makedirs(os.path.dirname(frontend_path), exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
    torch.onnx.export(
        actor,
        dummy,
        frontend_path,
        dynamo=False,
        opset_version=OPSET,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        do_constant_folding=True,
    )
    shutil.copyfile(frontend_path, checkpoint_path)


def make_obs(batch: int, obs_dim: int, rng: np.random.Generator) -> np.ndarray:
    realistic = rng.uniform(-1.0, 1.0, size=(batch, obs_dim))
    wide = rng.normal(0.0, 1.5, size=(batch, obs_dim))
    mask = rng.random((batch, 1)) < 0.5
    obs = np.where(mask, realistic, wide)
    return obs.astype(np.float32)


def parity_test(actor: Actor, onnx_path: str) -> bool:
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
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
        print(
            f"  batch={b:3d}  torch{torch_out.shape} vs onnx{ort_out.shape}  "
            f"max_abs_diff={max_diff:.3e}  {'PASS' if ok else 'FAIL'}"
        )

    print(
        f"\nONNX I/O: input='{in_name}' output='{out_name}' "
        f"(dynamic axis 0 = batch, feat dim = {actor.obs_dim} -> "
        f"act dim = {actor.act_dim})"
    )
    print(f"overall max_abs_diff = {overall_max:.3e}  (tol {PARITY_TOL:.0e})")
    print("PARITY: PASS" if all_pass else "PARITY: FAIL")
    return all_pass


def update_meta(meta_path: str, env_id: str, frontend_path: str, checkpoint_path: str) -> None:
    payload = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    payload["env_id"] = env_id
    payload["onnx"] = {
        "checkpoint": os.path.relpath(checkpoint_path, HERE),
        "frontend": os.path.relpath(frontend_path, REPO),
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="search-and-interdict")
    args = parser.parse_args()

    paths = checkpoint_paths(args.env_id)
    actor, ck = load_actor(paths["checkpoint"])
    print(
        f"[load] actor rebuilt from {paths['checkpoint']} "
        f"(env_id={ck.get('env_id', args.env_id)}, obs_dim={ck['obs_dim']}, "
        f"act_dim={ck['act_dim']}, hidden={ck['actor_hidden']})"
    )

    export(actor, ck["obs_dim"], paths["frontend"], paths["onnx"])
    print(f"[export] wrote {paths['frontend']} ({os.path.getsize(paths['frontend'])} bytes)")
    print(f"[export] copied  {paths['onnx']}")

    ok = parity_test(actor, paths["frontend"])
    update_meta(paths["meta"], args.env_id, paths["frontend"], paths["onnx"])
    if not ok:
        raise SystemExit("parity test FAILED — export not trustworthy")


if __name__ == "__main__":
    main()
