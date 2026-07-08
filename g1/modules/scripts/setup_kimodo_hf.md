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
On JetPack 5, the current upstream `bitsandbytes` path does not match this robot's
Python 3.8 / PyTorch 2.0 / CUDA 11.4 stack. The script can install CUDA PyTorch,
but current Kimodo's dependency tree also pulls Python 3.10+ packages. That means
JetPack 5 can be prepared for torch, but full Kimodo installation is not a
compatible on-robot path on this image.

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
   JetPack 5.x uses NVIDIA's direct Python 3.8 (`cp38`) torch wheel. JetPack 6.x
   can use the `pypi.jetson-ai-lab.io/jp6/...` indexes. The `jp6` devpi links do
   not help JetPack 5.

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

On this JetPack 5 robot, the script will use `~/.guv/envs/base` because NVIDIA's
JetPack 5 torch wheels are Python 3.8 wheels. It will not use
`~/.guv/envs/unitree` for torch unless torch is already installed there.

Skip quantization entirely and run the text encoder on CPU. This still requires
a compatible Kimodo/Python/torch stack, so it is not a workaround for JetPack 5's
Python 3.8 CUDA torch versus Kimodo Python 3.10+ dependency conflict:

```bash
./setup_kimodo_hf.bash --quantize cpu
```

Non-interactive (CI-style) run, no confirmation prompts:

```bash
./setup_kimodo_hf.bash --yes --run-smoke-test
```

Override the PyTorch wheel source:

```bash
./setup_kimodo_hf.bash --torch-wheel https://example.invalid/torch-cp38-linux_aarch64.whl
./setup_kimodo_hf.bash --index-url https://example.invalid/your-jp6-index
```

Install just PyTorch on JetPack 5 and stop before Kimodo/bitsandbytes:

```bash
./setup_kimodo_hf.bash --torch-only
```

Try the active Python 3.10 `unitree` environment with generic PyTorch instead
of NVIDIA's JetPack 5 CUDA wheel:

```bash
source ~/.guv/envs/unitree/bin/activate
./setup_kimodo_hf.bash --generic-torch --quantize cpu
```

This may satisfy Kimodo's Python 3.10+ dependency side, but generic PyTorch is
not the NVIDIA JetPack CUDA wheel. Check `torch.cuda.is_available()` before
expecting GPU acceleration.

On JetPack 5/aarch64 this mode installs base `kimodo`, not `kimodo[all]`.
The `all` extra pulls `py-soma-x`, which depends on `usd-core`; current
`usd-core` wheels are not published for aarch64 Python 3.10.
It also skips Kimodo's bundled `motion_correction` C++ extension because its
current CMake configuration uses x86-only SIMD flags on ARM.
The script installs Kimodo model/runtime dependencies and skips visualization
dependencies such as `scenepic` and Gradio, because `scenepic` currently fails
to build from sdist on this aarch64 Python 3.10 stack.

Full option list: `./setup_kimodo_hf.bash --help`.

## What the script does

1. Verifies it's running on a Jetson and detects L4T/JetPack version and CUDA
   toolkit version.
2. Reads total unified RAM and free disk, and prints an estimated peak-memory
   GO/NO-GO for the chosen `--quantize` mode (see table above for the
   estimates it uses).
3. Installs system packages via `apt-get` (`git`, `build-essential`, `cmake`,
   `ninja-build`, plus Jetson torch runtime libraries on JetPack 5) — prompts
   for confirmation before running (needs `sudo`).
4. Reuses a compatible virtual environment. On JetPack 5 that means Python 3.8
   (`~/.guv/envs/base` on this robot); on JetPack 6 that means Python 3.10.
   It installs `torch` from a Jetson-specific direct wheel or index, plus
   `bitsandbytes` from the configured index if a quantized mode was selected.
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

If you are using a text encoder service on another machine, force API mode so
Kimodo does not fall back to loading the local 8B Llama encoder:

```bash
TEXT_ENCODER_MODE=api TEXT_ENCODER_URL=http://HOST:9550/ \
  python3 kimodo_interactive.py --snapshot dev_stand_snapshot.json --no-robot
```

If the script used the onboard JetPack 5 `guv` base environment, activate that
environment for later runs:

```bash
source ~/.guv/envs/base/bin/activate
python kimodo_interactive.py --snapshot dev_stand_snapshot.json --no-robot
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

**`Selected Python is 3.10 ... JetPack 5 torch wheels require Python 3.8`** —
JetPack 5 NVIDIA torch wheels are `cp38`. Activate `~/.guv/envs/base`, or let
the script auto-select it. A `jp6` index cannot install a JetPack 5 CUDA wheel.

**`No torch wheel configured for this JetPack`** — pass a direct NVIDIA wheel
with `--torch-wheel`. `--index-url` is mainly for JetPack 6 indexes.

**`current Kimodo cannot be installed on this JetPack 5 Python 3.8 environment`** —
torch CUDA works on JetPack 5 only through NVIDIA's Python 3.8 (`cp38`) wheel,
but current Kimodo dependencies require Python 3.10+ packages. Use `--torch-only`,
upgrade to JetPack 6, or run Kimodo off-board on a workstation and send motions
to the robot.

**Using `--generic-torch`** — this keeps the active Python 3.10 environment and
installs generic PyTorch from the configured/default package index. It is useful
for testing Kimodo dependency resolution, but it is not expected to provide
JetPack 5 CUDA acceleration unless your index supplies a compatible aarch64 CUDA
build. On aarch64, avoid `--kimodo-extra all` because `py-soma-x`/`usd-core`
does not resolve for Python 3.10 on this platform. The script also sets
`SKIP_MOTION_CORRECTION_IN_SETUP=1` for this mode, so Kimodo postprocessing that
imports `motion_correction` will not be available. This path is intended for
`kimodo.model.load_model`, not the full Kimodo web demo/visualization stack.

**`bitsandbytes install failed`** — no prebuilt wheel exists for your
JetPack/CUDA combination on the community index. Current upstream
`bitsandbytes` requires newer Python/PyTorch/CUDA than JetPack 5 provides.
This is separate from the Kimodo Python-version conflict above.

**HF download returns 403 on `meta-llama/Meta-Llama-3-8B-Instruct`** — you
haven't accepted the Llama 3 license on huggingface.co with the account whose
token you're using. Accept it on the model page, then retry.

Check login state:

```bash
hf auth whoami
```

If it says `Not logged in`, create a read token at
`https://huggingface.co/settings/tokens`, accept the Llama 3 license at
`https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct`, then run:

```bash
hf auth login
```

The setup script now verifies access to the gated Llama config before reporting
that Hugging Face auth is ready.

**Generation runs but is extremely slow, or the process gets OOM-killed** —
the peak-memory estimates in this script are approximate; real usage depends
on prompt length, number of frames, and denoising steps. Reduce `--num-frames`
and `--steps` in `kimodo_interactive.py`, and check `dmesg` for OOM-killer
messages if the process dies without a Python traceback.

If startup prints `Text encoder service is unreachable, falling back to local
LLM2Vec encoder` and then the process is killed, no text encoder API was
reachable and Kimodo loaded the local Llama-3-8B encoder. Start
`python -m kimodo.scripts.run_text_encoder_server` on a machine with enough
RAM/VRAM and run with `TEXT_ENCODER_MODE=api TEXT_ENCODER_URL=http://HOST:9550/`,
or use a quantized local install and export `LLM2VEC_QUANTIZE=nf4` before
running.

If Kimodo reports that `kimodo.model.text_encoder_api.TextEncoderAPI` cannot be
located, make sure the active robot-side Python environment has the API client
dependency installed:

```bash
uv pip install gradio_client
```

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
