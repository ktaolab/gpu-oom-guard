#!/usr/bin/env python3
"""gpu-oom-guard: an early OOM killer for unified-memory NVIDIA systems.

On unified-memory machines (DGX Spark / GB10, Grace-based systems, Jetson) the
GPU allocates from system RAM through the NVIDIA kernel driver. Those pages are
not charged to any memory cgroup and the kernel OOM killer can't see or reclaim
them. A runaway GPU job therefore doesn't get killed: the machine thrashes swap
and hangs, often taking SSH/VPN daemons down with it.

gpu-oom-guard watches MemAvailable and memory pressure (PSI). When the host is
about to lock up it SIGKILLs the GPU process most likely to be at fault, chosen
by tier and then by GPU memory used. GPU usage is sampled while the system is
healthy, because nvidia-smi itself can block on the driver lock once memory is
gone.

Pure standard library, Python >= 3.10. Optional TOML config needs Python >= 3.11.
"""
from __future__ import annotations

import argparse
import ctypes
import fnmatch
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field

__version__ = "0.1.0"
GiB = 1024 ** 3

DEFAULT_PROTECTED = [
    "systemd", "Xorg", "Xwayland", "gnome-shell", "kwin_wayland", "kwin_x11",
    "sshd", "tailscaled", "nvidia-persistenced", "nvidia-smi",
]

# Tier rules, checked in order against the process's cgroup path (fnmatch).
# Lower tier = killed first. First match wins; unmatched -> default_tier.
DEFAULT_TIER_RULES = [
    # Containers and self-restarting services (containers are *.scope too, so first).
    ("*/docker-*", 1),
    ("*/docker/*", 1),
    ("*/libpod-*", 1),
    ("*/ollama.service", 1),
    # Interactive / ad-hoc work: shells, tmux, ssh sessions, desktop apps.
    ("*.scope", 0),
    ("*/session-*", 0),
    # Other services, system or user: killed last.
    ("*.service", 2),
]


@dataclass
class Config:
    warn_avail_gib: float = 12.0
    kill_avail_gib: float = 6.0
    kill_psi_full: float = 20.0
    hard_avail_gib: float = 3.0
    interval: float = 1.0
    sample_every: float = 5.0
    cooldown: float = 15.0
    sigterm_grace: float = 0.0
    dry_run: bool = False
    wall: bool = True
    default_tier: int = 0
    protected: list = field(default_factory=lambda: list(DEFAULT_PROTECTED))
    tier_rules: list = field(default_factory=lambda: list(DEFAULT_TIER_RULES))


# ---------------------------------------------------------------- system probes

def read_meminfo(path="/proc/meminfo") -> dict:
    d = {}
    with open(path) as f:
        for line in f:
            k, v = line.split(":", 1)
            d[k] = int(v.split()[0]) * 1024
    return d


def read_psi_full_avg10(path="/proc/pressure/memory") -> float:
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("full"):
                    for tok in line.split():
                        if tok.startswith("avg10="):
                            return float(tok[6:])
    except OSError:
        pass
    return 0.0


def parse_nvidia_smi(out: str) -> dict:
    res = {}
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            res[int(parts[0])] = int(parts[1]) * 1024 * 1024
        except ValueError:
            continue  # "[N/A]" etc.
    return res


def sample_gpu(timeout: float):
    """{pid: bytes} from nvidia-smi, or None if it is missing, fails or hangs."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout).stdout
    except (subprocess.TimeoutExpired, OSError):
        return None
    return parse_nvidia_smi(out)


@dataclass
class Proc:
    pid: int
    comm: str
    cgroup: str
    cmdline: str


def read_proc(pid: int, root="/proc"):
    try:
        with open(f"{root}/{pid}/comm") as f:
            comm = f.read().strip()
        with open(f"{root}/{pid}/cgroup") as f:
            # cgroup v2: "0::/path"; v1: several lines, take the systemd/name one or first
            lines = f.read().strip().splitlines()
        cg = next((l.split(":", 2)[2] for l in lines if l.startswith("0::")), None)
        if cg is None:
            cg = lines[0].split(":", 2)[2] if lines else ""
        with open(f"{root}/{pid}/cmdline", "rb") as f:
            cmd = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
        return Proc(pid, comm, cg, cmd[:300])
    except (OSError, IndexError):
        return None


# ---------------------------------------------------------------- policy

def tier_of(cgroup: str, cfg: Config) -> int:
    for pattern, tier in cfg.tier_rules:
        if fnmatch.fnmatch(cgroup, pattern):
            return int(tier)
    return cfg.default_tier


def rank_victims(usage: dict, cfg: Config, reader=read_proc) -> list:
    """Return [(tier, used_bytes, Proc)] in kill order."""
    cands = []
    for pid, used in usage.items():
        if pid <= 1 or pid == os.getpid():
            continue
        p = reader(pid)
        if p is None or p.comm in cfg.protected:
            continue
        cands.append((tier_of(p.cgroup, cfg), used, p))
    cands.sort(key=lambda c: (c[0], -c[1]))
    return cands


def in_danger(avail: int, psi: float, cfg: Config) -> bool:
    if avail < cfg.hard_avail_gib * GiB:
        return True
    return avail < cfg.kill_avail_gib * GiB and psi > cfg.kill_psi_full


# ---------------------------------------------------------------- daemon

def log(msg: str):
    print(msg, flush=True)


def wall(msg: str):
    if shutil.which("wall"):
        try:
            subprocess.run(["wall", msg], timeout=3, capture_output=True)
        except Exception:
            pass


def kill(pid: int, grace: float):
    try:
        if grace > 0:
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline:
                time.sleep(0.1)
                os.kill(pid, 0)
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def lock_memory():
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        if libc.mlockall(3) != 0:  # MCL_CURRENT | MCL_FUTURE
            log("gpu-oom-guard: warning: mlockall failed (errno %d); may be paged out under pressure"
                % ctypes.get_errno())
    except OSError:
        log("gpu-oom-guard: warning: libc not found, not locking memory")


def status(cfg: Config):
    m = read_meminfo()
    usage = sample_gpu(timeout=5) or {}
    print(f"MemAvailable {m['MemAvailable']/GiB:.1f} GiB, SwapFree {m.get('SwapFree', 0)/GiB:.1f} GiB, "
          f"PSI full avg10 {read_psi_full_avg10()}")
    print(f"thresholds: warn <{cfg.warn_avail_gib} GiB, kill <{cfg.kill_avail_gib} GiB with PSI "
          f">{cfg.kill_psi_full}, hard <{cfg.hard_avail_gib} GiB")
    ranked = rank_victims(usage, cfg)
    if not ranked:
        print("no killable GPU processes")
    for i, (t, used, p) in enumerate(ranked, 1):
        print(f"{i:2d}. pid {p.pid:<8} tier {t}  {used/GiB:6.1f} GiB  {p.comm:<16} {p.cgroup}")


def run(cfg: Config):
    if not cfg.dry_run:
        lock_memory()
    log(f"gpu-oom-guard {__version__}: warn<{cfg.warn_avail_gib}G kill<{cfg.kill_avail_gib}G"
        f"(psi>{cfg.kill_psi_full}) hard<{cfg.hard_avail_gib}G dry_run={cfg.dry_run}")
    if sample_gpu(timeout=5) is None:
        log("gpu-oom-guard: warning: nvidia-smi unavailable; will keep retrying")
    cache, cache_t, last_kill, last_warn = {}, -1e9, -1e9, -1e9
    while True:
        now = time.monotonic()
        avail = read_meminfo()["MemAvailable"]
        psi = read_psi_full_avg10()
        if now - cache_t >= cfg.sample_every:
            fresh = sample_gpu(timeout=3)
            if fresh is not None:
                cache, cache_t = fresh, now

        if in_danger(avail, psi, cfg):
            if now - last_kill >= cfg.cooldown:
                usage = sample_gpu(timeout=2) or cache
                ranked = rank_victims(usage, cfg)
                if ranked:
                    t, used, p = ranked[0]
                    verb = "WOULD KILL" if cfg.dry_run else "killing"
                    msg = (f"gpu-oom-guard: MemAvailable {avail/GiB:.1f} GiB, PSI {psi}: {verb} pid "
                           f"{p.pid} ({p.comm}, {used/GiB:.1f} GiB GPU, tier {t}, {p.cgroup}): {p.cmdline}")
                    log(msg)
                    if not cfg.dry_run:
                        kill(p.pid, cfg.sigterm_grace)
                        if cfg.wall:
                            wall(msg)
                    cache.pop(p.pid, None)
                else:
                    log(f"gpu-oom-guard: MemAvailable {avail/GiB:.1f} GiB but no killable GPU process")
                last_kill = now
        elif avail < cfg.warn_avail_gib * GiB and now - last_warn >= 60:
            log(f"gpu-oom-guard: warning: MemAvailable {avail/GiB:.1f} GiB, PSI {psi}")
            last_warn = now
        time.sleep(cfg.interval)


# ---------------------------------------------------------------- config / CLI

def load_toml(path: str, cfg: Config) -> Config:
    try:
        import tomllib
    except ImportError:
        sys.exit("gpu-oom-guard: TOML config needs Python >= 3.11")
    with open(path, "rb") as f:
        data = tomllib.load(f)
    for k, v in data.items():
        k = k.replace("-", "_")
        if k == "tier_rules":
            v = [(r["match"], int(r["tier"])) for r in v]
        elif k == "protected_extra":
            cfg.protected = cfg.protected + list(v)
            continue
        if not hasattr(cfg, k):
            sys.exit(f"gpu-oom-guard: unknown config key {k!r} in {path}")
        setattr(cfg, k, v)
    return cfg


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="gpu-oom-guard", description=__doc__.split("\n\n")[0])
    ap.add_argument("-c", "--config", help="TOML config file (default: /etc/gpu-oom-guard.toml if present)")
    ap.add_argument("--status", action="store_true", help="show memory state and kill order, then exit")
    ap.add_argument("--dry-run", action="store_true", help="log what would be killed, kill nothing")
    for name, help_ in [("warn-avail-gib", "log a warning below this MemAvailable"),
                        ("kill-avail-gib", "kill below this MemAvailable when PSI is high"),
                        ("kill-psi-full", "memory PSI 'full avg10' needed for a kill-avail kill"),
                        ("hard-avail-gib", "kill unconditionally below this MemAvailable"),
                        ("interval", "seconds between checks"),
                        ("sample-every", "seconds between nvidia-smi samples"),
                        ("cooldown", "minimum seconds between kills"),
                        ("sigterm-grace", "send SIGTERM and wait this long before SIGKILL (0 = SIGKILL now)")]:
        ap.add_argument("--" + name, type=float, help=help_)
    ap.add_argument("--version", action="version", version=__version__)
    return ap.parse_args(argv)


def build_config(a) -> Config:
    cfg = Config()
    path = a.config or ("/etc/gpu-oom-guard.toml" if os.path.exists("/etc/gpu-oom-guard.toml") else None)
    if path:
        load_toml(path, cfg)
    for k in ["warn_avail_gib", "kill_avail_gib", "kill_psi_full", "hard_avail_gib",
              "interval", "sample_every", "cooldown", "sigterm_grace"]:
        v = getattr(a, k)
        if v is not None:
            setattr(cfg, k, v)
    if a.dry_run:
        cfg.dry_run = True
    if cfg.hard_avail_gib > cfg.kill_avail_gib:
        sys.exit("gpu-oom-guard: hard_avail_gib must be <= kill_avail_gib")
    return cfg


def main(argv=None):
    a = parse_args(argv)
    cfg = build_config(a)
    if a.status:
        status(cfg)
        return
    try:
        run(cfg)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
