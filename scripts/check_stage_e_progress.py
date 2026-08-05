#!/usr/bin/env python3
"""Stage E formal training progress monitor.

Runs locally, SSHs to the GPU server, collects per-run progress for the
10 Stage-E formal runs (matd3/cop_maddpg x seeds 42,123,2026,3407,8888),
and writes a Markdown report to logs/stage_e_checks/.

Protocol (user convention): single SSH attempt with 8s timeout; on failure
write a "skipped" report and exit 0 -- the launchd schedule continues.

On detecting 10/10 completion, pulls new result files from the server
(/mnt/UAV/results/) into local results/stage_e_formal/ and flags the
pending analysis tasks in the report.

Designed to be run by launchd every 2 hours; safe to run manually anytime.
"""

import json
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paramiko  # noqa: E402

HOST = "172.16.30.53"
PORT = 22
USER = "root"
PASSWORD = "lingyi@2026"
SSH_TIMEOUT = 8

SERVER_LOG_DIR = "/root/Dyna-Q-DQN-UAV/logs"
SERVER_RESULTS_DIR = "/mnt/UAV/results"
LOG_GLOB = "*stagee_conservative*.log"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(PROJECT_ROOT, "logs", "stage_e_checks")
PULL_DIR = os.path.join(PROJECT_ROOT, "results", "stage_e_formal")
STATE_FILE = os.path.join(REPORT_DIR, "state.json")

EXPECTED = {f"{algo}_seed{seed}" for algo in ("matd3", "cop_maddpg")
            for seed in (42, 123, 2026, 3407, 8888)}

RE_EP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - \S+ - INFO - "
                   r"Ep\s+(\d+) \| avg_r\(last_50\)=\s*([-\d.]+) \| best_ra=\s*(\S+)")
RE_DONE = re.compile(r"completed: (\d+) episodes, duration=([\d.]+)s, "
                     r"stopped_early=(\w+), reason=(.*)")
RE_NAME = re.compile(r"(\w+?)_paper_xi_stagee_conservative_seed(\d+)_")
ANOMALY_KEYS = ("Traceback", "NaN", "nan", "CUDA error", "out of memory", "OOM")


def log(msg):
    print(msg, flush=True)


def ssh_connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD,
                   timeout=SSH_TIMEOUT, banner_timeout=SSH_TIMEOUT,
                   auth_timeout=SSH_TIMEOUT)
    return client


def run_cmd(client, cmd, timeout=30):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    return stdout.read().decode(errors="replace")


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"runs": {}, "pulled_files": [], "history": []}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)


def parse_run(key, log_text):
    """Parse one run's log text into a status dict."""
    info = {"key": key, "status": "unknown", "episode": None,
            "avg_r50": None, "best_ra": None, "last_ts": None,
            "episodes_total": None, "duration_s": None,
            "stopped_early": None, "reason": None, "anomalies": []}
    for line in log_text.splitlines():
        m = RE_EP.search(line)
        if m:
            info["status"] = "running"
            info["last_ts"] = m.group(1)
            info["episode"] = int(m.group(2))
            info["avg_r50"] = float(m.group(3))
            info["best_ra"] = m.group(4)
        d = RE_DONE.search(line)
        if d:
            info["status"] = "done"
            info["episodes_total"] = int(d.group(1))
            info["duration_s"] = float(d.group(2))
            info["stopped_early"] = d.group(3)
            info["reason"] = d.group(4).strip()
        for k in ANOMALY_KEYS:
            if k in line and "INFO" not in line:
                info["anomalies"].append(line.strip()[:200])
    if info["status"] == "unknown":
        info["status"] = "no-progress-lines"
    return info


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(PULL_DIR, exist_ok=True)
    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M")
    state = load_state()

    lines = [f"# 阶段 E 正式训练定时检查报告",
             f"",
             f"- 本地检查时间：{now.strftime('%Y-%m-%d %H:%M')}",
             f"- 检查方式：launchd 每 2 小时（单次 SSH 尝试，超时 8s 即跳过）",
             f""]

    try:
        client = ssh_connect()
    except Exception as e:
        lines += [f"## 本轮跳过",
                  f"",
                  f"SSH 连接失败（{type(e).__name__}: {e}）。按约定不重试，",
                  f"下一轮按计划继续。训练在服务器上不受影响。"]
        report_path = os.path.join(REPORT_DIR, f"report_{stamp}_skipped.md")
        with open(report_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        log(f"[skipped] SSH failed: {e}\n-> {report_path}")
        return

    try:
        # --- collect raw data ---
        procs = run_cmd(client, "ps aux | grep run_stage_e_formal | grep -v grep")
        n_workers = len([l for l in procs.splitlines() if l.strip()])
        gpu = run_cmd(client, "nvidia-smi --query-gpu=index,utilization.gpu,"
                              "memory.used --format=csv,noheader,nounits")
        listing = run_cmd(client, f"ls -t {SERVER_LOG_DIR}/{LOG_GLOB} 2>/dev/null")
        log_files = [l.strip() for l in listing.splitlines() if l.strip()]
        results_listing = run_cmd(client, f"ls -t {SERVER_RESULTS_DIR} | head -10")

        runs = {}
        for path in log_files:
            name = RE_NAME.search(os.path.basename(path))
            if not name:
                continue
            key = f"{name.group(1)}_seed{name.group(2)}"
            tail = run_cmd(client, f"tail -40 '{path}'")
            collisions = run_cmd(client, f"grep -ci collision '{path}'").strip()
            info = parse_run(key, tail)
            info["collision_lines"] = collisions
            info["log"] = path
            runs[key] = info

        # --- rates & ETA from state ---
        prev = state.get("runs", {})
        now_iso = now.isoformat()
        for key, info in runs.items():
            p = prev.get(key)
            if p and info["episode"] and p.get("episode") is not None \
                    and info["status"] == "running" and p.get("ts"):
                try:
                    dt_h = (now - datetime.fromisoformat(p["ts"])).total_seconds() / 3600
                    de = info["episode"] - p["episode"]
                    if dt_h > 0.1 and de >= 0:
                        info["rate_eph"] = round(de / dt_h, 1)
                except Exception:
                    pass
            state["runs"][key] = {"episode": info["episode"], "ts": now_iso,
                                  "status": info["status"]}

        done = [k for k, v in runs.items() if v["status"] == "done"]
        running = [k for k, v in runs.items() if v["status"] == "running"]
        queued = sorted(EXPECTED - set(runs.keys()))

        # reference stop episodes per algo for ETA
        ref_stop = {}
        for k in done:
            algo = k.split("_seed")[0]
            ref_stop.setdefault(algo, []).append(runs[k]["episodes_total"])

        # --- report ---
        lines += [f"## 总进度：{len(done)}/10 完成",
                  f"",
                  f"- 服务器 worker 进程数：{n_workers}",
                  f"- GPU 状态：`{gpu.strip().replace(chr(10), ' ; ')}`",
                  f""]
        lines += ["| 运行 | 状态 | Episode | avg_r(50) | best_ra | 速率(ep/h) | 预计剩余 | 碰撞日志行 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for key in sorted(EXPECTED):
            info = runs.get(key)
            if info is None:
                lines.append(f"| {key} | 排队 | — | — | — | — | — | — |")
                continue
            if info["status"] == "done":
                lines.append(f"| {key} | ✅ 完成 | {info['episodes_total']} "
                             f"({info['stopped_early'] == 'True' and '早停' or '跑满'}) | — | — | — "
                             f"| {info['duration_s']/3600:.1f}h | {info['collision_lines']} |")
            else:
                eta = "—"
                algo = key.split("_seed")[0]
                if info.get("rate_eph") and ref_stop.get(algo) and info["episode"]:
                    ref = sum(ref_stop[algo]) / len(ref_stop[algo])
                    remain_h = max(0.0, (ref - info["episode"]) / info["rate_eph"])
                    eta = f"~{remain_h:.1f}h（参考停止 {ref:.0f} 轮）"
                lines.append(f"| {key} | 🔄 Ep {info['episode']} | {info['episode']} "
                             f"| {info['avg_r50']} | {info['best_ra']} "
                             f"| {info.get('rate_eph', '—')} | {eta} | {info['collision_lines']} |")

        anomaly_lines = [(k, a) for k, v in runs.items() for a in v["anomalies"]]
        lines += ["", "## 异常扫描", ""]
        if n_workers == 0 and len(done) < 10:
            lines.append(f"- ⚠️ **worker 进程为 0 但仅 {len(done)}/10 完成——主进程可能已退出，需人工核查**")
        if anomaly_lines:
            for k, a in anomaly_lines[:10]:
                lines.append(f"- ⚠️ {k}: `{a}`")
        if not anomaly_lines and not (n_workers == 0 and len(done) < 10):
            lines.append("- 未发现 Traceback / NaN / CUDA OOM。")
        lines += ["", "## 服务器结果目录（最新 10 项）", "",
                  "```", results_listing.strip(), "```", ""]

        # --- completion: pull results ---
        if len(done) == 10:
            pulled = set(state.get("pulled_files", []))
            # only files created after the Stage-E formal launch (2026-08-03 16:00
            # server time), plus the formal manifest dir
            remote_new = run_cmd(
                client, f"find {SERVER_RESULTS_DIR} -maxdepth 1 -type f "
                        f"-newermt '2026-08-03 16:00' "
                        f"\\( -name 'benchmark_report_*' -o -name '*stage_e*' \\) "
                        f"-printf '%f\\n'").split()
            manifest_ls = run_cmd(
                client, f"ls {SERVER_RESULTS_DIR}/stage_e_formal/ 2>/dev/null").split()
            remote_new += [f"stage_e_formal/{f}" for f in manifest_ls if f.strip()]
            new_files = [f for f in remote_new if f not in pulled]
            if new_files:
                sftp = client.open_sftp()
                got = []
                for f in new_files:
                    try:
                        local_path = os.path.join(PULL_DIR, f)
                        os.makedirs(os.path.dirname(local_path), exist_ok=True)
                        sftp.get(f"{SERVER_RESULTS_DIR}/{f}", local_path)
                        got.append(f)
                    except Exception as e:
                        log(f"[warn] pull {f} failed: {e}")
                sftp.close()
                state["pulled_files"] = sorted(pulled | set(got))
                lines += ["## ✅ 10/10 全部完成", "",
                          f"已拉取 {len(got)} 个结果文件到 `results/stage_e_formal/`：", ""]
                lines += [f"- `{g}`" for g in got]
                lines += ["", "**待办（下次会话）**：Ξ 对照分析 → 更新 wcl/ 主表 → "
                              "写阶段 E 正式报告。", ""]
            else:
                lines += ["## ✅ 10/10 全部完成（结果文件此前已拉取）", ""]

        report_path = os.path.join(REPORT_DIR, f"report_{stamp}.md")
        text = "\n".join(lines) + "\n"
        with open(report_path, "w") as f:
            f.write(text)

        summary = (f"| {now.strftime('%m-%d %H:%M')} | {len(done)}/10 | "
                   f"{len(running)} 运行 | {len(queued)} 排队 | worker={n_workers} |")
        hist = os.path.join(REPORT_DIR, "history.md")
        if not os.path.exists(hist):
            with open(hist, "w") as f:
                f.write("| 时间 | 完成 | 运行中 | 排队 | worker |\n|---|---|---|---|---|\n")
        with open(hist, "a") as f:
            f.write(summary + "\n")

        state["history"].append({"ts": now_iso, "done": len(done)})
        save_state(state)
        log(text)
        log(f"-> {report_path}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
