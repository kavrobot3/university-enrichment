#!/usr/bin/env python3
"""
University enrichment - one pipeline, two interchangeable providers.

Shared by OmniRoute and OpenRouter (run one at a time):
  data/universities.csv    source, never modified
  output/output.csv        ONLY rows with every enrichment field filled
  output/repair_queue.csv  partial/failed rows; keeps filled fields, retries only missing ones
  output/checkpoint.json   metadata: schema, per-provider model health, last run
  output/enrich.log        size-capped log

  python enrich.py --provider omniroute  --run
  python enrich.py --provider openrouter --run
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
ORIGINAL_COLUMNS = ["name", "country", "domains", "web_pages", "alpha_two_code", "state-province"]
PROMPT_COLUMNS = ["name", "country", "state-province", "web_pages"]
BLANKISH = {"", "null", "none", "nil", "unknown", "n/a", "na", "not available",
            "not applicable", "-", "insufficient reliable information"}
DAILY_LIMIT = ("per-day", "per day", "daily limit", "daily quota")
WRITE_LOCK = threading.Lock()

DEFAULT_SYSTEM = (
    "You enrich university records. Reply with ONE JSON object containing exactly the "
    "requested keys. Every value is a non-empty string of at most {max_words} words.\n"
    "- State what you reliably know plainly.\n"
    "- If unsure, still answer: start the value with 'AI estimate:' and give a brief basis "
    "(country, institution type, comparable institutions).\n"
    "- Never invent published rankings, ranking sources, emails, phone numbers or URLs.\n"
    "- Never output null, empty strings, 'unknown' or 'N/A'."
)


# ------------------------------------------------------------------ utilities

def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_time(s: Any):
    try:
        t = datetime.fromisoformat(str(s))
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def to_int(v: Any) -> int:
    try:
        return int(float(str(v).strip()))
    except Exception:
        return 0


def is_meaningful(v: Any) -> bool:
    return str(v or "").strip().lower().strip(" .") not in BLANKISH


def row_key(row: dict) -> str:
    return "||".join(str(row.get(k, "")).strip().lower()
                     for k in ("name", "country", "domains", "web_pages"))


def first_url(web_pages: Any) -> str:
    s = str(web_pages or "")
    m = re.search(r"https?://[^\s,;'\"\[\]]+", s)
    return m.group(0) if m else s.strip(" []'\"")


def merge_list(existing: Any, *new: str) -> str:
    out = [x.strip() for x in str(existing or "").split(",") if x.strip()]
    for n in new:
        if n and n not in out:
            out.append(n)
    return ",".join(out)


def read_csv_flexible(path: Path) -> pd.DataFrame:
    last = None
    for enc in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, dtype=str, keep_default_na=False)
        except Exception as e:
            last = e
    raise RuntimeError(f"Could not read {path}: {last}")


def _retry_io(fn, what: str):
    for i in range(10):
        try:
            return fn()
        except PermissionError:
            logging.warning("%s is locked (open in Excel?) - retrying in 3s", what)
            time.sleep(3)
    return fn()


def append_row(path: Path, row: dict, columns: list[str]) -> None:
    def do():
        path.parent.mkdir(parents=True, exist_ok=True)
        new = (not path.is_file()) or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8-sig" if new else "utf-8") as f:
            w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({c: str(row.get(c, "") or "") for c in columns})
    _retry_io(do, path.name)


def write_csv_atomic(path: Path, rows: list[dict], columns: list[str]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: str(r.get(c, "") or "") for c in columns})
    _retry_io(lambda: os.replace(tmp, path), path.name)


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except Exception:
        return {}


def setup_logging(cfg: dict) -> None:
    log = ROOT / cfg["log_file"]
    log.parent.mkdir(parents=True, exist_ok=True)
    cap = float(cfg.get("log_max_mb", 5)) * 1024 * 1024
    if log.is_file() and log.stat().st_size > cap:
        log.write_bytes(log.read_bytes()[-int(cap * 0.2):])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(log, encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])
    for noisy in ("httpx", "openai", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ------------------------------------------------------------------ config

def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    fields = cfg.get("enrichment_fields") or []
    names = [f.get("name") for f in fields]
    if not fields or not all(names) or len(set(names)) != len(names):
        sys.exit("config.json: enrichment_fields need unique names")
    for f in fields:
        if not f.get("description"):
            sys.exit(f"config.json: field needs a description: {f}")
        f.setdefault("type", "string")
    if not cfg.get("providers"):
        sys.exit("config.json: no providers block")
    return cfg


def field_names(cfg: dict) -> list[str]:
    return [f["name"] for f in cfg["enrichment_fields"]]


def output_columns(cfg: dict) -> list[str]:
    return ["_source_row"] + ORIGINAL_COLUMNS + field_names(cfg) + ["_provider", "_model_used"]


def queue_columns(cfg: dict) -> list[str]:
    return (["_source_row"] + ORIGINAL_COLUMNS + field_names(cfg)
            + ["_missing_fields", "_attempts", "_last_error", "_queue_reason",
               "_provider", "_model_used", "_updated_at"])


def schema_fingerprint(cfg: dict) -> str:
    payload = [{"name": f["name"], "type": f["type"]} for f in cfg["enrichment_fields"]]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def pick_provider(cfg: dict, name: str | None) -> tuple[str, dict]:
    name = name or cfg.get("active_provider")
    if name not in cfg["providers"]:
        sys.exit(f"Unknown provider {name!r}. Available: {', '.join(cfg['providers'])}")
    return name, cfg["providers"][name]


def model_list(pcfg: dict) -> list[str]:
    models = [str(m).strip() for m in (pcfg.get("models") or []) if str(m).strip()]
    if not models:
        sys.exit("Provider has no models configured")
    return models


def make_client(pcfg: dict) -> OpenAI:
    load_dotenv(ROOT / ".env", override=True)
    var = str(pcfg.get("api_key_env") or "").strip()
    key = os.environ.get(var, "").strip().strip('"').strip("'")
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    if not key:
        sys.exit(f"{var} is not set. Add it to .env (no quotes).")
    return OpenAI(base_url=str(pcfg["base_url"]).rstrip("/"), api_key=key,
                  default_headers={"HTTP-Referer": "https://localhost/university-enrichment",
                                   "X-Title": "University Enrichment"},
                  timeout=float(pcfg.get("timeout_seconds", 180)), max_retries=0)


# ------------------------------------------------------------------ prompting

def build_messages(uni: dict, cfg: dict, fields: list[str]) -> list[dict]:
    desc = {f["name"]: f["description"] for f in cfg["enrichment_fields"]}
    system = str(cfg.get("system_prompt") or DEFAULT_SYSTEM).replace(
        "{max_words}", str(cfg.get("max_words_per_field", 50)))
    record = {c: str(uni.get(c, "")).strip() for c in PROMPT_COLUMNS if str(uni.get(c, "")).strip()}
    user = ("Record: " + json.dumps(record, ensure_ascii=False)
            + "\nReturn JSON with keys: " + ", ".join(fields) + "\n"
            + "\n".join(f"- {n}: {desc[n]}" for n in fields))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_json(text: str) -> Any:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    if not text:
        raise ValueError("empty model response")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def parse_values(raw: Any, wanted: list[str]) -> dict[str, str]:
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        raw = raw[0]
    if not isinstance(raw, dict):
        raise ValueError(f"expected JSON object, got {type(raw).__name__}")
    lower = {str(k).strip().lower().replace(" ", "_"): v for k, v in raw.items()}
    out = {}
    for name in wanted:
        v = raw.get(name, lower.get(name.lower()))
        if isinstance(v, list):
            v = "; ".join(str(x) for x in v if str(x).strip())
        elif isinstance(v, dict):
            v = "; ".join(f"{k}: {x}" for k, x in v.items())
        s = re.sub(r"\s+", " ", "" if v is None else str(v)).strip()
        if is_meaningful(s):
            out[name] = s
    return out


# ------------------------------------------------------------------ transport

def status_of(exc: Exception):
    s = getattr(exc, "status_code", None)
    if s is None and getattr(exc, "response", None) is not None:
        s = getattr(exc.response, "status_code", None)
    return s


def classify(exc: Exception, cfg: dict) -> str:
    """'dead' = rotate model, 'wait' = rate limit, 'retry' = ordinary error."""
    fo = cfg.get("model_failover") or {}
    enabled = fo.get("enabled", True)
    st, msg = status_of(exc), str(exc).lower()
    if enabled and type(exc).__name__ == "APIConnectionError":
        return "dead"
    if enabled and st in (401, 402, 403, 404):
        return "dead"
    if enabled and any(p in msg for p in DAILY_LIMIT):
        return "dead"
    if st == 429 or "rate limit" in msg or "too many requests" in msg:
        return "wait"
    if enabled and any(str(p).lower() in msg for p in fo.get("rotate_on_patterns") or []):
        return "dead"
    return "retry"


class JsonMode:
    def __init__(self, mode: str):
        self.mode = (mode or "auto").lower()

    def enabled(self) -> bool:
        return self.mode in ("on", "auto")

    def disable(self) -> None:
        if self.mode == "auto":
            self.mode = "off"
            logging.info("response_format=json rejected by endpoint; disabled for this run")


class ModelPool:
    def __init__(self, models: list[str], state: dict, revive_hours: float, max_streak: int):
        self.models, self.max_streak = models, max_streak
        self.lock = threading.RLock()
        self.dead: dict[str, str] = {}
        self.streak: Counter = Counter()
        raw = state.get("dead_models") or {}
        if isinstance(raw, list):
            raw = {m: utc_now() for m in raw}
        cutoff = datetime.now(timezone.utc) - timedelta(hours=float(revive_hours or 0))
        for m, ts in raw.items():
            if m not in models:
                continue
            t = parse_time(ts)
            if revive_hours and t and t < cutoff:
                logging.info("Reviving %s (marked dead %s)", m, ts)
                continue
            self.dead[m] = ts

    def current(self):
        with self.lock:
            return next((m for m in self.models if m not in self.dead), None)

    def mark_dead(self, model: str, reason: str) -> None:
        with self.lock:
            if model in self.models and model not in self.dead:
                self.dead[model] = utc_now()
                logging.error("FAILOVER: %s unavailable (%s)", model, reason)
                nxt = self.current()
                logging.info("FAILOVER: now using %s", nxt) if nxt else \
                    logging.error("FAILOVER: every model for this provider is unavailable")

    def record(self, model: str, ok: bool) -> None:
        with self.lock:
            if ok:
                self.streak[model] = 0
                return
            self.streak[model] += 1
            if self.max_streak and self.streak[model] >= self.max_streak:
                self.mark_dead(model, f"{self.streak[model]} consecutive errors")

    def snapshot(self) -> dict:
        with self.lock:
            return {"active_model": self.current(), "dead_models": dict(self.dead)}


def complete_once(client: OpenAI, pcfg: dict, model: str, messages: list, jm: JsonMode):
    kw: dict[str, Any] = {"model": model, "messages": messages,
                          "temperature": float(pcfg.get("temperature", 0))}
    if pcfg.get("max_tokens"):
        kw["max_tokens"] = int(pcfg["max_tokens"])
    if pcfg.get("extra_body"):
        kw["extra_body"] = pcfg["extra_body"]
    if jm.enabled():
        try:
            r = client.chat.completions.create(**kw, response_format={"type": "json_object"})
        except Exception as e:
            if status_of(e) in (400, 415, 422) or "response_format" in str(e).lower():
                jm.disable()
                r = client.chat.completions.create(**kw)
            else:
                raise
    else:
        r = client.chat.completions.create(**kw)
    if not getattr(r, "choices", None):
        raise ValueError(f"no choices in response: {getattr(r, 'error', '')}")
    ch = r.choices[0]
    return (ch.message.content or ""), getattr(ch, "finish_reason", None)


# ------------------------------------------------------------------ repair queue

class RepairQueue:
    def __init__(self, path: Path, columns: list[str]):
        self.path, self.columns, self.rows = path, columns, {}

    def load(self, done_keys: set) -> "RepairQueue":
        self.rows = {}
        if self.path.is_file() and self.path.stat().st_size:
            for r in read_csv_flexible(self.path).to_dict("records"):
                k = row_key(r)
                if k not in done_keys:          # last occurrence wins
                    self.rows[k] = {c: str(r.get(c, "") or "") for c in self.columns}
        return self

    def upsert(self, row: dict) -> None:
        with WRITE_LOCK:
            self.rows[row_key(row)] = row
            append_row(self.path, row, self.columns)

    def write(self) -> None:                     # caller holds WRITE_LOCK or no workers run
        rows = sorted(self.rows.values(), key=lambda r: to_int(r.get("_source_row")))
        write_csv_atomic(self.path, rows, self.columns)

    def compact(self) -> None:
        with WRITE_LOCK:
            self.write()


# ------------------------------------------------------------------ state

def source_records(cfg: dict) -> list[dict]:
    path = ROOT / cfg["input_csv"]
    if not path.is_file():
        sys.exit(f"Missing input CSV: {path}")
    df = read_csv_flexible(path)
    miss = [c for c in ORIGINAL_COLUMNS if c not in df.columns]
    if miss:
        sys.exit(f"Input CSV missing columns: {', '.join(miss)}")
    recs = df.to_dict("records")
    for i, r in enumerate(recs, start=1):
        r["_source_row"] = str(i)
    return recs


def load_state(cfg: dict):
    recs = source_records(cfg)
    out = ROOT / cfg["output_csv"]
    done = set()
    if out.is_file() and out.stat().st_size:
        done = {row_key(r) for r in read_csv_flexible(out).to_dict("records")}
    queue = RepairQueue(ROOT / cfg["queue_csv"], queue_columns(cfg)).load(done)
    return recs, done, queue


def build_pending(cfg: dict, args, recs: list, done: set, queue: RepairQueue, scope: str):
    fields = field_names(cfg)
    total = len(recs)
    hi = int(args.to_record or cfg.get("end_record") or total)
    q_lo = int(args.from_record or 1)
    f_lo = int(args.from_record or cfg.get("start_record") or 1)
    max_q = int(cfg.get("max_queue_attempts", 4))
    items, info = [], {"queue": 0, "fresh": 0, "parked": 0, "f_lo": f_lo, "hi": hi}
    if scope in ("all", "queue"):
        for row in queue.rows.values():
            n = to_int(row.get("_source_row"))
            if not q_lo <= n <= hi:
                continue
            if max_q and to_int(row.get("_attempts")) >= max_q:
                info["parked"] += 1
                continue
            items.append(dict(row))
            info["queue"] += 1
    if scope in ("all", "fresh"):
        for r in recs:
            n = int(r["_source_row"])
            k = row_key(r)
            if f_lo <= n <= hi and k not in done and k not in queue.rows:
                items.append({**r, **{f: "" for f in fields}})
                info["fresh"] += 1
    items.sort(key=lambda r: to_int(r.get("_source_row")))
    return items, info


def check_output_header(path: Path, cols: list[str]) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        return
    with path.open(encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f), [])
    if header != cols:
        sys.exit(f"{path.name} header does not match config.\n  file  : {header}\n  config: {cols}")


def save_checkpoint(cfg: dict, pname: str, pool: ModelPool, stats: dict, batch: int, batches: int):
    path = ROOT / cfg["checkpoint_json"]
    ck = load_json(path)
    ck["updated_at"] = utc_now()
    ck["schema"] = schema_fingerprint(cfg)
    ck.setdefault("providers", {})[pname] = {**pool.snapshot(), "last_run_at": utc_now()}
    ck["last_run"] = {"provider": pname, "batch": batch, "batches": batches, **stats}
    write_json(path, ck)


# ------------------------------------------------------------------ worker

class Ctx:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def enrich_fields(ctx: Ctx, uni: dict, wanted: list[str]):
    """Ask only for `wanted`; re-ask only for what is still missing.
    Returns (values, models_used, last_error, halt) where halt is None|'exhausted'|'stopped'."""
    cfg = ctx.cfg
    remaining, got, used = list(wanted), {}, set()
    attempts = waits = 0
    last_err = None
    max_attempts = max(1, int(cfg.get("max_attempts_per_row", 3)))
    sleeps = list(cfg.get("retry_sleep_seconds") or [2, 4, 8])
    max_waits = int(cfg.get("max_rate_limit_waits", 6))
    cooldown = float((cfg.get("model_failover") or {}).get("cooldown_seconds", 2))
    n = uni.get("_source_row", "")

    while remaining and attempts < max_attempts:
        if ctx.stop.is_set():
            return got, used, last_err, "stopped"
        model = ctx.pool.current()
        if model is None:
            return got, used, last_err, "exhausted"
        attempts += 1
        try:
            text, finish = complete_once(ctx.client, ctx.pcfg, model,
                                         build_messages(uni, cfg, remaining), ctx.jm)
            try:
                raw = extract_json(text)
            except Exception:
                if finish == "length":
                    raise ValueError("reply truncated (finish_reason=length) - raise max_tokens")
                raise
            vals = parse_values(raw, remaining)
            ctx.pool.record(model, True)
            if vals:
                got.update(vals)
                used.add(model)
                remaining = [f for f in remaining if f not in got]
            else:
                last_err = ValueError("reply had no usable values for " + ",".join(remaining))
            if remaining and attempts < max_attempts:
                logging.info("#%s still missing %s - re-asking for those only", n, ",".join(remaining))
        except Exception as e:
            last_err = e
            kind = classify(e, cfg)
            short = str(e).replace("\n", " ")[:200]
            if kind == "dead":
                ctx.pool.mark_dead(model, short)
                attempts -= 1
                time.sleep(cooldown)
                continue
            if kind == "wait":
                if waits < max_waits:
                    waits += 1
                    delay = min(10 * waits, 90) + random.uniform(0, 3)
                    logging.warning("#%s rate-limited on %s; waiting %.0fs", n, model, delay)
                    time.sleep(delay)
                    attempts -= 1
                    continue
                ctx.pool.mark_dead(model, f"still rate-limited after {waits} waits")
                attempts -= 1
                continue
            ctx.pool.record(model, False)
            logging.warning("#%s attempt %s/%s on %s: %s", n, attempts, max_attempts, model, short)
            if attempts < max_attempts:
                time.sleep(sleeps[min(attempts - 1, len(sleeps) - 1)])
    return got, used, last_err, None


def process_item(ctx: Ctx, item: dict) -> str:
    if ctx.stop.is_set():
        return "skip"
    try:
        fields, cfg = ctx.fields, ctx.cfg
        n, name = item.get("_source_row", ""), item.get("name", "")
        key = row_key(item)
        values = {f: (item.get(f, "") if is_meaningful(item.get(f)) else "") for f in fields}
        wanted = [f for f in fields if not values[f]]
        got, used, err, halt = (enrich_fields(ctx, item, wanted) if wanted
                                else ({}, set(), None, None))
        values.update(got)

        if (halt is None and "contact" in fields and not values.get("contact")
                and cfg.get("contact_fallback_to_source_website", True)):
            url = first_url(item.get("web_pages"))
            if url:
                values["contact"] = f"General website (from source record): {url}"

        missing = [f for f in fields if not values[f]]
        providers = merge_list(item.get("_provider"), ctx.pname if used else "")
        models = merge_list(item.get("_model_used"), *sorted(used))

        if not missing:
            row = {"_source_row": n, **{c: item.get(c, "") for c in ORIGINAL_COLUMNS},
                   **values, "_provider": providers, "_model_used": models}
            with WRITE_LOCK:
                append_row(ctx.out_path, row, ctx.out_cols)
                ctx.done_keys.add(key)
                ctx.queue.rows.pop(key, None)
            logging.info("OK      #%s %s | filled=%s", n, name, ",".join(wanted) or "-")
            return "ok"

        if halt == "exhausted":
            ctx.stop.set()
        if halt and not got:
            return "stop" if halt == "exhausted" else "skip"

        qrow = {"_source_row": n, **{c: item.get(c, "") for c in ORIGINAL_COLUMNS}, **values,
                "_missing_fields": ",".join(missing),
                "_attempts": str(to_int(item.get("_attempts")) + (0 if halt else 1)),
                "_last_error": str(err or "").replace("\n", " ")[:300],
                "_queue_reason": item.get("_queue_reason") or "incomplete after model attempts",
                "_provider": providers, "_model_used": models, "_updated_at": utc_now()}
        ctx.queue.upsert(qrow)
        if halt:
            return "stop" if halt == "exhausted" else "skip"
        logging.info("QUEUED  #%s %s | still missing=%s", n, name, ",".join(missing))
        return "queued"
    except Exception as e:
        logging.error("ERROR   #%s %s | %s", item.get("_source_row", ""), item.get("name", ""), e)
        return "error"
    finally:
        if ctx.sleep_s > 0:
            time.sleep(ctx.sleep_s)


# ------------------------------------------------------------------ commands

def run(cfg: dict, pname: str, pcfg: dict, args, test_batch: bool = False) -> None:
    fields = field_names(cfg)
    out_path = ROOT / cfg["output_csv"]
    check_output_header(out_path, output_columns(cfg))
    ck = load_json(ROOT / cfg["checkpoint_json"])
    if ck.get("schema") and ck["schema"] != schema_fingerprint(cfg):
        sys.exit("enrichment_fields names/types changed vs existing output. Revert config.")

    recs, done, queue = load_state(cfg)
    items, info = build_pending(cfg, args, recs, done, queue, args.scope)
    bs = int(cfg.get("batch_size", 50))
    if test_batch:
        items = items[:bs]
    logging.info("provider=%s pending=%s (queue=%s fresh=%s, parked=%s) up to record %s",
                 pname, len(items), info["queue"], info["fresh"], info["parked"], info["hi"])
    if not items:
        logging.info("Nothing to do.")
        return

    if args.dry_run:
        it = items[0]
        wanted = [f for f in fields if not is_meaningful(it.get(f))]
        msgs = build_messages(it, cfg, wanted)
        for m in msgs:
            print(f"--- {m['role']} ---\n{m['content']}\n")
        print(f"Record #{it['_source_row']} asks for: {', '.join(wanted)}")
        print(f"~{sum(len(m['content']) for m in msgs) // 4} input tokens. "
              f"Would process {len(items)} rows. No API calls made.")
        return

    queue.compact()   # normalise columns before any appends
    pool = ModelPool(model_list(pcfg), (ck.get("providers") or {}).get(pname, {}),
                     cfg.get("dead_model_revive_hours", 6), int(cfg.get("max_consecutive_errors", 10)))
    if pool.current() is None:
        sys.exit(f"All {pname} models are marked unavailable. Use the other provider, "
                 f"wait {cfg.get('dead_model_revive_hours', 6)}h, or run --reset-models.")

    ctx = Ctx(cfg=cfg, pcfg=pcfg, pname=pname, client=make_client(pcfg), pool=pool,
              jm=JsonMode(pcfg.get("json_mode", "auto")), queue=queue, done_keys=done,
              fields=fields, out_path=out_path, out_cols=output_columns(cfg),
              stop=threading.Event(), sleep_s=float(cfg.get("request_sleep_seconds", 0.2)))
    stats: Counter = Counter()
    batches = [items[i:i + bs] for i in range(0, len(items), bs)]
    ex = ThreadPoolExecutor(max_workers=max(1, int(cfg.get("concurrency", 1))))
    interrupted, bi = False, 0
    try:
        for bi, batch in enumerate(batches, start=1):
            if ctx.stop.is_set():
                break
            logging.info("=== batch %s/%s | records %s-%s | model=%s ===", bi, len(batches),
                         batch[0]["_source_row"], batch[-1]["_source_row"], pool.current())
            pend = {ex.submit(process_item, ctx, it) for it in batch}
            while pend:
                try:
                    finished, pend = wait(pend, timeout=1.0)
                except KeyboardInterrupt:
                    if interrupted:
                        raise
                    interrupted = True
                    ctx.stop.set()
                    for f in pend:
                        f.cancel()
                    logging.warning("Ctrl+C: finishing in-flight requests so paid replies are "
                                    "saved. Press Ctrl+C again to quit immediately.")
                    continue
                for f in finished:
                    if f.cancelled():
                        continue
                    try:
                        stats[f.result()] += 1
                    except Exception as e:
                        stats["error"] += 1
                        logging.error("worker error: %s", e)
            queue.compact()
            save_checkpoint(cfg, pname, pool, dict(stats), bi, len(batches))
            logging.info("batch %s saved | ok=%s queued=%s errors=%s | complete=%s queue=%s",
                         bi, stats["ok"], stats["queued"], stats["error"], len(done), len(queue.rows))
    except KeyboardInterrupt:
        with WRITE_LOCK:
            try:
                queue.write()
                save_checkpoint(cfg, pname, pool, dict(stats), bi, len(batches))
            except Exception as e:
                print(f"save on exit failed: {e}")
            logging.warning("Stopped immediately; in-flight replies discarded.")
            logging.shutdown()
            os._exit(130)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

    if pool.current() is None:
        other = next((p for p in cfg["providers"] if p != pname), "other")
        logging.warning("All %s models unavailable. Progress saved. Continue with:  "
                        "python enrich.py --provider %s --run", pname, other)
    logging.info("Run finished: %s", dict(stats))


def cmd_status(cfg: dict, args) -> None:
    recs, done, queue = load_state(cfg)
    items, info = build_pending(cfg, args, recs, done, queue, args.scope)
    fields = field_names(cfg)
    combos = Counter(",".join(f for f in fields if not is_meaningful(r.get(f)))
                     for r in queue.rows.values())
    ck = load_json(ROOT / cfg["checkpoint_json"])
    print(f"Source records          : {len(recs)}")
    print(f"Complete (output.csv)   : {len(done)}")
    print(f"Repair queue            : {len(queue.rows)}  "
          f"(parked after {cfg.get('max_queue_attempts', 4)} attempts: {info['parked']})")
    print(f"Fresh, never tried      : {info['fresh']}  (records {info['f_lo']}-{info['hi']})")
    print(f"Pending in this range   : {len(items)}")
    print(f"Next record to process  : {items[0]['_source_row'] if items else '-'}")
    print("Queue by missing fields :")
    for combo, c in combos.most_common():
        print(f"    {combo or '(none - will move to output)':<32} {c}")
    print(f"Last checkpoint         : {ck.get('updated_at', '-')}")
    for p in cfg["providers"]:
        st = (ck.get("providers") or {}).get(p, {})
        print(f"  {p:<11} active={st.get('active_model', '-')}  dead={st.get('dead_models', {})}")


def cmd_audit(cfg: dict) -> None:
    fields = field_names(cfg)
    out_path = ROOT / cfg["output_csv"]
    out = read_csv_flexible(out_path) if out_path.is_file() and out_path.stat().st_size else pd.DataFrame()
    print(f"output.csv rows: {len(out)}")
    for f in fields:
        if f not in out.columns:
            print(f"  {f:<11} MISSING COLUMN")
            continue
        col = out[f].astype(str)
        print(f"  {f:<11} blank={int((~col.map(is_meaningful)).sum()):<5} "
              f"labelled-estimate={int(col.str.contains(r'estimat', case=False).sum())}")
    if "contact" in out.columns:
        print(f"  contact from source-website fallback: "
              f"{int(out['contact'].astype(str).str.startswith('General website').sum())}")
    if "_provider" in out.columns and len(out):
        print("  rows by provider:", out["_provider"].value_counts().head(8).to_dict())
    bad = sum(int(out[f].astype(str).str.contains("û").sum()) for f in fields if f in out.columns)
    print(f"  cells containing 'û' (encoding check, expect ~0): {bad}")


def cmd_ping(pname: str, pcfg: dict) -> None:
    client, ok = make_client(pcfg), False
    for model in model_list(pcfg):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0, max_tokens=int(pcfg.get("max_tokens", 1024)),
                messages=[{"role": "user", "content": 'Reply with exactly {"ok": true}'}])
            print(f"OK   [{pname}] {model}: {(r.choices[0].message.content or '').strip()[:100]}")
            ok = True
        except Exception as e:
            print(f"FAIL [{pname}] {model}: {str(e)[:300]}")
    if not ok:
        sys.exit("No working model for this provider.")


def cmd_list_models(pcfg: dict) -> None:
    ids = sorted(m.id for m in make_client(pcfg).models.list().data)
    print(f"{len(ids)} models:")
    for m in ids:
        print(" ", m)


def cmd_reset_models(cfg: dict, pname: str) -> None:
    path = ROOT / cfg["checkpoint_json"]
    ck = load_json(path)
    ck.setdefault("providers", {}).setdefault(pname, {})["dead_models"] = {}
    write_json(path, ck)
    print(f"Cleared unavailable-model list for {pname}.")


def main() -> None:
    p = argparse.ArgumentParser(description="University enrichment")
    p.add_argument("--config", default="config.json")
    p.add_argument("--provider")
    p.add_argument("--from-record", dest="from_record", type=int)
    p.add_argument("--to-record", dest="to_record", type=int)
    p.add_argument("--scope", choices=["all", "queue", "fresh"], default="all")
    p.add_argument("--dry-run", action="store_true")
    g = p.add_mutually_exclusive_group()
    for flag in ("run", "test-batch", "status", "audit", "ping", "list-models", "reset-models"):
        g.add_argument(f"--{flag}", action="store_true")
    args = p.parse_args()

    cfg_path = ROOT / args.config
    if not cfg_path.is_file():
        sys.exit(f"Missing {cfg_path}")
    cfg = load_config(cfg_path)
    setup_logging(cfg)
    pname, pcfg = pick_provider(cfg, args.provider)

    if args.status:
        cmd_status(cfg, args)
    elif args.audit:
        cmd_audit(cfg)
    elif args.ping:
        cmd_ping(pname, pcfg)
    elif args.list_models:
        cmd_list_models(pcfg)
    elif args.reset_models:
        cmd_reset_models(cfg, pname)
    elif args.test_batch:
        run(cfg, pname, pcfg, args, test_batch=True)
    elif args.run:
        run(cfg, pname, pcfg, args)
    else:
        print("Use --status, --audit, --ping, --list-models, --test-batch, --run, --reset-models")


if __name__ == "__main__":
    main()