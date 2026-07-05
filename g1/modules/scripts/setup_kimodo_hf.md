# Kimodo + Hugging Face setup on the G1 EDU onboard Jetson

`setup_kimodo_hf.bash` installs [Kimodo](https://github.com/nv-tlabs/kimodo) (NVIDIA's
text-to-motion diffusion model, used by `kimodo_interactive.py` / `kimodo_replay.py`
in this directory) directly on the G1 EDU's onboard Jetson, including its Hugging
Face model downloads.

**Read the "Hardware reality" section before running this.** Kimodo was built and
tested on desktop GPUs (RTX 3090/4090, A100) with dedicated VRAM. The G1 EDU's
onboard Jetson has none of that — running Kimodo on it at all is a memory-fitting
exercise, not a drop-in install, and this script is written to make that fitting
problem visible rather than hide it behind defaults that quietly fail.

## Hardware reality

Jetson modules use **unified memory** — the GPU and CPU share the same LPDDR RAM,
there is no separate VRAM pool. The G1 EDU's onboard compute is a Jetson Orin NX
module (16 GB variant per Unitree's spec sheet) on a custom Unitree carrier board.

Kimodo's memory cost is almost entirely its text encoder, not the motion model
itself (the diffusion transformer is only ~282M parameters). By default Kimodo
uses `meta-llama/Meta-Llama-3-8B-Instruct` as that text encoder:

| Config | Text encoder | Approx. peak memory | Fits in 16 GB unified? |
|---|---|---|---|
| Official default (`--quantize none`) | Llama-3-8B fp16 on GPU | ~17 GB | No |
| `TEXT_ENCODER_DEVICE=cpu` (`--quantize cpu`) | Llama-3-8B fp16 on **CPU RAM** | ~17 GB (moved to system RAM instead of VRAM) | No — same total, wrong pool |
| `LLM2VEC_QUANTIZE=int8` | Llama-3-8B int8 | ~9 GB | Maybe, thin margin |
| `LLM2VEC_QUANTIZE=nf4` / `fp4` | Llama-3-8B NF4/FP4 | ~5–6 GB | Most plausible option |

The int8/nf4/fp4 modes come from a third-party fork,
[matbeedotcom/kimodo](https://github.com/matbeedotcom/kimodo), which adds
`bitsandbytes` quantization of the text encoder — **it is not official NVIDIA
code**, and neither that fork nor `bitsandbytes` documents Jetson/aarch64 support.
The script installs a Jetson-built `bitsandbytes` wheel if one is available on the
[jetson-ai-lab](https://pypi.jetson-ai-lab.io/) community index; if it isn't,
quantization can't proceed without building `bitsandbytes` from source yourself.

None of the numbers above include the RAM your robot's ROS/DDS control stack is
already using on that same Jetson. `--quantize nf4` (the script's default) is the
only mode with a realistic chance of leaving headroom for that — and even it is
tight. Run `./setup_kimodo_hf.bash --check-only` first and read its output before
deciding whether to proceed.

**If you don't need it running on the robot itself**, the more reliable option is
still to run Kimodo on a workstation with a real GPU and let `kimodo_interactive.py`
publish `rt/lowcmd` over the network to the robot — nothing about Kimodo requires
it to run on the onboard Jetson specifically. Use this on-Jetson setup only if
standalone/tetherless operation is a hard requirement.

## Prerequisites

1. **Run this on the Jetson itself** (SSH into the robot's onboard computer),
   not your dev workstation. The script checks for `/etc/nv_tegra_release` /
   `/proc/device-tree/model` and refuses to run anywhere else.
2. **Internet access from the Jetson** — for apt, the Jetson PyTorch wheel index,
   and Hugging Face downloads (several GB either way; see the table above).
3. **A Hugging Face account**, with:
   - A [User Access Token](https://huggingface.co/settings/tokens) (read scope
     is enough).
   - The [Meta Llama 3 license](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
     accepted on that same account — Llama-3-8B-Instruct is gated, and Kimodo's
     download will fail with a 403 until you've accepted it on the HF website.
4. **~40 GB free disk** as a starting point (default `--min-free-gb 40`); more if
   you use `--quantize cpu`/`none`, which pull the full fp16 Llama-3-8B weights.
5. Know your **JetPack version** ahead of time if possible (`cat /etc/nv_tegra_release`).
   The script currently only has a wheel-index mapping for **JetPack 6.x**
   (via `pypi.jetson-ai-lab.io/jp6/cu126`). JetPack 5.x Jetsons need a manual
   `--index-url` — the community index no longer publishes `jp5` wheels; you'd
   need NVIDIA's official per-version wheel from the
   [Jetson PyTorch forum thread](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
   instead.

## Usage

Check hardware feasibility without installing anything:

```bash
./setup_kimodo_hf.bash --check-only
```

This prints detected JetPack/L4T version, CUDA version, total unified RAM, free
disk, and a GO/NO-GO estimate for the default `nf4` quantization mode.

Full install with the default (most likely to fit) quantization mode:

```bash
./setup_kimodo_hf.bash --run-smoke-test
```

Skip quantization entirely and run the text encoder on CPU (only realistic if
you have more than 16 GB of unified memory, e.g. a future 32 GB Jetson variant):

```bash
./setup_kimodo_hf.bash --quantize cpu
```

Non-interactive (CI-style) run, no confirmation prompts:

```bash
./setup_kimodo_hf.bash --yes --run-smoke-test
```

Override the PyTorch wheel index (e.g. you found a `jp5` wheel manually):

```bash
./setup_kimodo_hf.bash --index-url https://example.invalid/your-jp5-index
```

Full option list: `./setup_kimodo_hf.bash --help`.

## What the script does

1. Verifies it's running on a Jetson and detects L4T/JetPack version and CUDA
   toolkit version.
2. Reads total unified RAM and free disk, and prints an estimated peak-memory
   GO/NO-GO for the chosen `--quantize` mode (see table above for the
   estimates it uses).
3. Installs system packages via `apt-get` (`git`, `build-essential`, `cmake`,
   `ninja-build`, `python3.10`, `python3.10-venv`) — prompts for confirmation
   before running (needs `sudo`). Kimodo requires Python 3.10; on JetPack 5
   (Python 3.8 by default) it offers to add the `deadsnakes` PPA.
4. Creates a venv (default `~/kimodo_venv`) and installs `torch`/`torchvision`
   from the Jetson-specific wheel index, then `bitsandbytes` from the same
   index if a quantized mode was selected.
5. `pip install`s `kimodo[all]` — from the official `nv-tlabs/kimodo` repo for
   `--quantize cpu|none`, or the `matbeedotcom/kimodo` quantization fork for
   `--quantize nf4|fp4|int8`.
6. Checks for an existing Hugging Face token and offers to run `hf auth login`
   if none is found. Reminds you to accept the Llama-3 license first.
7. Sets `LLM2VEC_QUANTIZE` or `TEXT_ENCODER_DEVICE` for the current shell
   session (add whichever applies to your shell profile to persist it).
8. With `--run-smoke-test`, loads the model and generates one short
   (~1 second) test motion from a fixed prompt, printing the output tensor
   shape — this is the first real signal that the whole chain (weights
   downloaded, encoder quantized/placed correctly, GPU usable) actually works
   on this device.

Every apt-get run, every multi-GB download, and the `hf auth login` step are
gated behind a confirmation prompt unless you pass `-y`/`--yes`.

## After setup: running a real prompt

The smoke test above is a minimal sanity check. To actually generate and review
a motion using the same safety analysis and joint mapping the robot scripts use
(without touching the robot):

```bash
source ~/kimodo_venv/bin/activate
python3 kimodo_interactive.py --snapshot dev_stand_snapshot.json --no-robot
```

`--no-robot` runs Kimodo generation and the full safety analysis (limit
violations, ramp warnings, high-velocity frames) and stops at the replay
confirmation — nothing is sent to the robot. Drop `--no-robot` (and confirm
the developer-mode prompt) once you've reviewed generated motions and are
ready to actually replay one on the hanging robot.

## Troubleshooting

**`This does not look like a Jetson`** — you ran the script somewhere other
than the onboard Jetson (e.g. your workstation). SSH into the robot's onboard
computer first.

**`No known Jetson PyTorch wheel index for this JetPack version`** — you're on
JetPack 5.x (L4T R35) or older. The `jetson-ai-lab` community index currently
only serves JetPack 6 wheels. Find a matching official NVIDIA wheel on the
[Jetson PyTorch forum thread](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
for your exact JetPack/CUDA version and pass it with `--index-url`.

**`bitsandbytes install failed`** — no prebuilt wheel exists for your
JetPack/CUDA combination on the community index. Building `bitsandbytes` from
source for Jetson's compute capability is not automated by this script.
Re-run with `--quantize cpu` to skip quantization (only advisable if you have
enough unified RAM — see the table above), or investigate a source build.

**HF download returns 403 on `meta-llama/Meta-Llama-3-8B-Instruct`** — you
haven't accepted the Llama 3 license on huggingface.co with the account whose
token you're using. Accept it on the model page, then retry.

**Generation runs but is extremely slow, or the process gets OOM-killed** —
the peak-memory estimates in this script are approximate; real usage depends
on prompt length, number of frames, and denoising steps. Reduce `--num-frames`
and `--steps` in `kimodo_interactive.py`, and check `dmesg` for OOM-killer
messages if the process dies without a Python traceback.

**Robot control gets sluggish or drops out while Kimodo is loaded** — this is
the unified-memory/CPU-contention tradeoff described above: the onboard Jetson
is also running the robot's control stack. If this happens, treat it as
confirmation that on-Jetson generation isn't a good fit for this hardware, and
switch to running Kimodo on a workstation instead (see "Hardware reality").

## Sources

- [nv-tlabs/kimodo](https://github.com/nv-tlabs/kimodo) — official Kimodo repo
- [Kimodo installation docs](https://research.nvidia.com/labs/sil/projects/kimodo/docs/getting_started/installation.html)
- [nvidia/Kimodo-G1-SEED-v1 on Hugging Face](https://huggingface.co/nvidia/Kimodo-G1-SEED-v1)
- [matbeedotcom/kimodo](https://github.com/matbeedotcom/kimodo) — bitsandbytes quantization fork (third-party, unofficial)
- [pypi.jetson-ai-lab.io](https://pypi.jetson-ai-lab.io/) — community Jetson wheel index (PyTorch, bitsandbytes)
