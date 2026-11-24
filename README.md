# gpu-oom-guard

An early OOM killer for **unified-memory NVIDIA systems** (DGX Spark / GB10, Grace-based machines, Jetson): it kills the right GPU job *before* the machine hangs.

## The problem

On unified-memory machines the GPU allocates from system RAM through the NVIDIA kernel driver. Those pages:

- aren't charged to any memory cgroup, so `MemoryMax=`, Docker `--memory` and systemd-oomd don't limit or see them;
- aren't attributed to the process, so the kernel OOM killer can't pick (or free) the culprit;
- can't be swapped.

So when GPU jobs together ask for more than the machine has, nothing gets killed. The kernel swaps out every *other* process trying to make room, the box thrashes, and it eventually stops responding. Remote-access daemons (SSH, Tailscale) usually die first, so you lose the machine until someone power-cycles it.

A real incident this was written after, on a 128 GB GB10:

| time | event |
|---|---|
| T+0 | a batch-inference script starts on the GPU next to an image-generation server, a TTS server and Ollama |
| T+4 min | a scheduled job makes Ollama load a second 21 GB model; the driver starts logging `NV_ERR_NO_MEMORY` |
| T+14 min | 98.5% of RAM in use: **65 MB** anonymous memory, 0.6 GB page cache, the rest driver allocations. Swap thrashing. The kernel OOM killer never fires. |
| T+32 min | the VPN daemon's watchdog trips and it exits; the host is unreachable |
| T+49 min | manual power cycle |

## What it does

A small root daemon (Python standard library only, one file) that every second:

1. reads `MemAvailable` and memory pressure (`/proc/pressure/memory`, "full avg10");
2. every few seconds, while the system is healthy, samples per-process GPU memory with `nvidia-smi`. The sample is cached because `nvidia-smi` itself blocks on the driver lock once memory is exhausted;
3. when memory is nearly gone, **SIGKILLs one GPU process**, chosen by tier and then by size.

Default trigger: `MemAvailable < 6 GiB` with PSI full > 20 (the system is actually stalling), or `MemAvailable < 3 GiB` regardless. After a kill it waits 15 s before killing again.

### Who gets killed

Processes are tiered by their cgroup, lowest tier first, largest GPU user first within a tier:

| tier | default match | typically |
|---|---|---|
| 0 | `*.scope`, `*/session-*` | things you started by hand: tmux panes, SSH sessions, notebooks, ad-hoc scripts, coding-agent shells |
| 1 | `*/docker-*`, `*/libpod-*`, `*/ollama.service` | containers and model servers that restart themselves |
| 2 | `*.service` | other long-running services |

The reasoning: the job that tips a shared box over is usually the newest, hand-started one, and services with `Restart=` come back on their own. Desktop and remote-access processes (`Xorg`, `gnome-shell`, `sshd`, `tailscaled`, ...) are never killed. Rules, protected names and thresholds are all configurable.

## Install

Requires Linux with systemd, cgroup v2, PSI (kernel ≥ 4.20, on by default on Ubuntu) and `nvidia-smi`.

```bash
git clone https://github.com/ktaolab/gpu-oom-guard
cd gpu-oom-guard
sudo ./install.sh                    # guard only
sudo ./install.sh --with-hardening   # guard + the extras below
```

Check it:

```bash
sudo gpu-oom-guard --status          # memory state and current kill order
journalctl -u gpu-oom-guard -f       # warnings and kills
```

Try it without risk: `gpu-oom-guard --dry-run --hard-avail-gib 999` logs what it *would* kill right now.

Uninstall: `sudo ./install.sh --uninstall`.

### `--with-hardening`

Optional extras that keep the machine reachable, or bring it back, if the guard isn't enough:

- **tailscaled / sshd drop-ins:** `Restart=always`, `OOMScoreAdjust=-900`, `MemoryMin=256M`. Applied to the running daemons without restarting them, so an SSH or Tailscale SSH session running the installer stays up.
- **Hardware watchdog** (if `/dev/watchdog` exists): `RuntimeWatchdogSec=2min`, so a fully hung kernel reboots itself.
- **sysctl:** `vm.min_free_kbytes=1GiB`, `vm.watermark_scale_factor=100`, to start reclaim earlier and keep a reserve. Scale to about 1% of RAM.

Each file is in [`hardening/`](hardening/), so you can install them by hand instead.

## Configuration

Copy [`gpu-oom-guard.example.toml`](gpu-oom-guard.example.toml) to `/etc/gpu-oom-guard.toml` (needs Python ≥ 3.11 for TOML; CLI flags work on Python ≥ 3.10) and `systemctl restart gpu-oom-guard`.

Example: make a long-running render or training service the last thing to go:

```toml
kill_avail_gib = 8

[[tier_rules]]
match = "*/docker-*"
tier = 1
[[tier_rules]]
match = "*/render-server.service"
tier = 3
[[tier_rules]]
match = "*.scope"
tier = 0
[[tier_rules]]
match = "*.service"
tier = 2
```

Setting `tier_rules` replaces the defaults, and the first match wins, so list specific rules before general ones.

## Also worth doing

The guard is the safety net. These reduce how often you need it:

- **Score only the last token.** For classification or likelihood scoring with Hugging Face models, `model(**enc, logits_to_keep=1)` avoids materialising `batch × seq × vocab` logits (≈10 GB per batch for batch 8 × 4096 tokens × a 152k vocab).
- **Cap PyTorch:** `torch.cuda.set_per_process_memory_fraction(0.4)`.
- **Bound Ollama:** `OLLAMA_MAX_LOADED_MODELS` and `OLLAMA_NUM_PARALLEL` multiply memory use; Ollama's own free-memory check runs only at load time.

## Limitations

- Tested on DGX Spark (GB10, Ubuntu 24.04, kernel 6.17, driver 580+). It should work on other unified-memory NVIDIA systems where `nvidia-smi --query-compute-apps` reports per-process memory. Reports welcome.
- On discrete-GPU machines VRAM exhaustion doesn't hang the host, so you don't need this there.
- It kills without asking. A long training run that is the biggest ad-hoc GPU user when memory runs out will die. Checkpoint, or move it to a higher tier.
- If `nvidia-smi` has never answered since the guard started, it has no GPU data and only logs the danger.

## License

MIT
