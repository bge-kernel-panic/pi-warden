# Step 02 — PyTorch on the Radeon 7800 XT via ROCm

**Where:** the home Debian box, with the 7800 XT installed.

**Key fact that makes this easy:** ROCm PyTorch keeps the **CUDA API names**.
`torch.cuda.is_available()`, `device="cuda"`, `torch.autocast("cuda")` all work
unchanged on AMD — HIP pretends to be CUDA underneath. So we do **not** rewrite
device calls in the notebook; we just need a PyTorch built for ROCm.

You have two ways to get that. **The Docker container is strongly recommended** —
it avoids ROCm/PyTorch version-matching pain entirely. Do bare-metal only if you
already run ROCm workloads on this box and prefer it.

---

## Option A (recommended): the ROCm PyTorch container

### A1. Confirm the host can see the GPU

```bash
rocm-smi
```

You should see the 7800 XT listed with a temperature/power reading. If `rocm-smi`
is missing, ROCm is not installed on the host — install ROCm from AMD's Debian
instructions first (that is a prerequisite this plan assumes, since you said ROCm
already works on this card).

### A2. Pull and run the container

```bash
docker pull rocm/pytorch:latest

docker run -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --ipc=host --shm-size 16G \
  --security-opt seccomp=unconfined \
  -v "$HOME/src/pi-warden":/work \
  -w /work/training \
  rocm/pytorch:latest bash
```

Notes:
- `--device` / `--group-add` expose the GPU to the container.
- `--shm-size 16G` matters: dataloaders crash with tiny shared memory.
- `-v .../pi-warden:/work` mounts this repo so the container sees `training/data`
  and the notebook. Adjust the path if your checkout lives elsewhere.

**Everything from here on (step 03, step 04) runs *inside* this container shell.**

### A3. Verify PyTorch sees the GPU (inside the container)

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda avail', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0))"
```

Expect:
```
torch 2.x.x+rocm6.x
cuda avail True
device  <something like 'AMD Radeon RX 7800 XT' or a gfx name>
```

`cuda avail True` is the thing that must be true. If it is `False`, see
Troubleshooting below.

---

## Option B (bare metal): pip install ROCm PyTorch

Inside a fresh venv on the host:

```bash
python -m venv ~/laya-venv && source ~/laya-venv/bin/activate
pip install --upgrade pip
# Match the ROCm version installed on your host. Check `ls /opt/rocm*` for it.
# Example for ROCm 6.2 — change the URL suffix to match yours:
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
```

Then run the same A3 verify command. If `torch.__version__` does not end in
`+rocmX.Y`, you installed the CUDA build by mistake — uninstall and reinstall from
the rocm index URL.

---

## Troubleshooting `cuda avail False`

The 7800 XT is `gfx1101` (Navi 32). Some ROCm builds only ship kernels for the
top-tier `gfx1100` (7900 XTX). The standard workaround is to tell ROCm to treat
the card as `gfx1100`:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

Re-run the A3 verify. If it now prints `True`, add that `export` to your shell
profile (and pass it into the container with `-e HSA_OVERRIDE_GFX_VERSION=11.0.0`
on the `docker run` line). You said ROCm already works on this card, so you may
not need this — but it is the first thing to try if the GPU is invisible.

Also useful:
- `export HIP_VISIBLE_DEVICES=0` — force the first GPU if you have more than one.
- `rocminfo | grep gfx` — shows the actual gfx arch the driver reports.

---

**When A3 prints `cuda avail True` with the 7800 XT named, step 02 is done.**
Move to `03-port-notebook.md` (still inside the container shell).
