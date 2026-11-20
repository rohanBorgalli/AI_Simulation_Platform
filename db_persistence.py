# db_persistence.py
# Hardened Supabase PostgREST + Local JSON Fallback Persistence Layer
# Zero Extra Dependencies (Uses only Python standard library + `requests`)
# Developed by Dr. Rohan Borgalli, Shah & Anchor Kutchhi Engineering College, Chembur

import os
import json
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import requests

try:
    import streamlit as st
except ImportError:
    st = None

APP_PREFIX = "sakec_simlab"
USERS_TABLE = f"{APP_PREFIX}_users"
RECORDS_TABLE = f"{APP_PREFIX}_records"

LOCAL_USERS_FILE = f"{APP_PREFIX}_users.json"
LOCAL_RECORDS_FILE = f"{APP_PREFIX}_records.json"


def _sanitize_value(obj: Any) -> Any:
    """Recursively converts NumPy scalars/arrays, sets, datetimes, and non-serializable objects into JSON-safe primitives."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    # Handle NumPy scalar types without importing numpy explicitly unless present
    type_mod = type(obj).__module__
    if "numpy" in type_mod:
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _sanitize_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_value(v) for v in obj]
    return str(obj)


def _normalize_id(raw_id: str) -> str:
    """Normalizes primary keys (usernames/emails/composite IDs) to lowercase trimmed strings."""
    return str(raw_id).strip().lower()


def _normalize_role(raw_role: str) -> str:
    """Normalizes roles to lowercase ('admin' or 'user')."""
    r = str(raw_role or "user").strip().lower()
    return r if r in ("admin", "user") else "user"


def _warn_ui(msg: str) -> None:
    """Surfaces clear non-silent warnings in Streamlit UI and prints to server logs."""
    print(f"[DB_PERSISTENCE WARNING] {msg}")
    if st is not None:
        try:
            st.warning(f"⚠️ Cloud Persistence Notice: {msg}")
        except Exception:
            pass


def _get_supabase_credentials() -> Tuple[Optional[str], Optional[str]]:
    """
    Reads SUPABASE_URL and SUPABASE_KEY from environment variables OR st.secrets
    (supporting both top-level keys and nested [supabase] TOML section).
    """
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_KEY", "").strip()

    if (not url or not key) and st is not None:
        try:
            # 1. Check top-level keys in st.secrets
            if not url and "SUPABASE_URL" in st.secrets:
                url = str(st.secrets["SUPABASE_URL"]).strip()
            if not key and "SUPABASE_KEY" in st.secrets:
                key = str(st.secrets["SUPABASE_KEY"]).strip()

            # 2. Check nested [supabase] section in st.secrets
            if (not url or not key) and "supabase" in st.secrets:
                sec = st.secrets["supabase"]
                if not url:
                    url = str(sec.get("SUPABASE_URL") or sec.get("url") or "").strip()
                if not key:
                    key = str(sec.get("SUPABASE_KEY") or sec.get("key") or sec.get("anon_key") or "").strip()
        except Exception:
            pass

    if url and key:
        return url.rstrip("/"), key
    return None, None


class SupabasePostgRESTStore:
    """
    Zero-dependency PostgREST client with paginated GETs, batched 200-row atomic upserts,
    JSON sanitization, and automatic local JSON fallback.
    """

    def __init__(self):
        self.seeded_tables = set()

    def _headers(self, key: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        hdrs = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        if extra:
            hdrs.update(extra)
        return hdrs

    def _read_local_json(self, filepath: str) -> Dict[str, Dict[str, Any]]:
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    return raw
            except Exception as e:
                _warn_ui(f"Failed reading local fallback file `{filepath}`: {e}")
        return {}

    def _write_local_json(self, filepath: str, mapping: Dict[str, Dict[str, Any]]) -> None:
        try:
            clean_map = _sanitize_value(mapping)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(clean_map, f, indent=2, ensure_ascii=False)
        except Exception as e:
            _warn_ui(f"Failed writing local fallback file `{filepath}`: {e}")

    def fetch_all(self, table: str, local_fallback_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Fetches all rows from Supabase using paginated Range headers (0-999, 1000-1999, ...)
        so datasets over 1,000 rows are never truncated. Falls back to local JSON if offline.
        """
        url, key = _get_supabase_credentials()
        if not url or not key:
            return self._read_local_json(local_fallback_path)

        endpoint = f"{url}/rest/v1/{table}?select=id,data,updated_at&order=updated_at.asc"
        all_rows: List[Dict[str, Any]] = []
        page_size = 1000
        offset = 0

        try:
            while True:
                range_header = f"{offset}-{offset + page_size - 1}"
                resp = requests.get(
                    endpoint,
                    headers=self._headers(key, {"Range": range_header, "Range-Unit": "items"}),
                    timeout=15
                )
                if resp.status_code not in (200, 206):
                    _warn_ui(
                        f"Supabase GET `{table}` failed (HTTP {resp.status_code}): {resp.text[:250]}. "
                        f"Using local JSON fallback."
                    )
                    return self._read_local_json(local_fallback_path)

                batch = resp.json()
                if not isinstance(batch, list):
                    break
                all_rows.extend(batch)
                if len(batch) < page_size:
                    break
                offset += page_size

            # One-time auto-seed if Supabase table is empty and local JSON file exists
            if len(all_rows) == 0 and table not in self.seeded_tables:
                self.seeded_tables.add(table)
                local_seed = self._read_local_json(local_fallback_path)
                if local_seed:
                    self.upsert_many(table, local_seed, local_fallback_path)
                    return local_seed

            result_map: Dict[str, Dict[str, Any]] = {}
            for row in all_rows:
                rec_id = _normalize_id(row.get("id", ""))
                rec_data = row.get("data") or {}
                if rec_id and isinstance(rec_data, dict):
                    result_map[rec_id] = rec_data

            # Mirror latest cloud snapshot to local JSON cache
            if result_map:
                self._write_local_json(local_fallback_path, result_map)
            return result_map

        except Exception as e:
            _warn_ui(f"Supabase GET `{table}` exception: {e}. Using local JSON fallback.")
            return self._read_local_json(local_fallback_path)

    def upsert_one(self, table: str, record_id: str, data_payload: Dict[str, Any], local_fallback_path: str) -> bool:
        """
        Atomic single-record upsert (`POST /rest/v1/<table>?on_conflict=id` with
        `Prefer: resolution=merge-duplicates,return=minimal`) so concurrent users never overwrite each other.
        """
        norm_id = _normalize_id(record_id)
        clean_data = _sanitize_value(data_payload)
        now_iso = datetime.now(timezone.utc).isoformat()

        # Always update local JSON cache
        local_map = self._read_local_json(local_fallback_path)
        local_map[norm_id] = clean_data
        self._write_local_json(local_fallback_path, local_map)

        url, key = _get_supabase_credentials()
        if not url or not key:
            return True

        endpoint = f"{url}/rest/v1/{table}?on_conflict=id"
        body = [{"id": norm_id, "data": clean_data, "updated_at": now_iso}]
        try:
            resp = requests.post(
                endpoint,
                headers=self._headers(key, {"Prefer": "resolution=merge-duplicates,return=minimal"}),
                data=json.dumps(body),
                timeout=15
            )
            if resp.status_code not in (200, 201, 204):
                _warn_ui(f"Supabase UPSERT `{table}` (id={norm_id}) failed (HTTP {resp.status_code}): {resp.text[:250]}")
                return False
            return True
        except Exception as e:
            _warn_ui(f"Supabase UPSERT `{table}` (id={norm_id}) exception: {e}")
            return False

    def upsert_many(self, table: str, records_map: Dict[str, Dict[str, Any]], local_fallback_path: str) -> bool:
        """
        Chunks bulk uploads into batches of 200 rows using atomic merge-duplicates upserts.
        """
        if not records_map:
            return True

        now_iso = datetime.now(timezone.utc).isoformat()
        sanitized_rows = []
        local_map = self._read_local_json(local_fallback_path)

        for raw_id, payload in records_map.items():
            norm_id = _normalize_id(raw_id)
            clean_data = _sanitize_value(payload)
            local_map[norm_id] = clean_data
            sanitized_rows.append({"id": norm_id, "data": clean_data, "updated_at": now_iso})

        self._write_local_json(local_fallback_path, local_map)

        url, key = _get_supabase_credentials()
        if not url or not key:
            return True

        endpoint = f"{url}/rest/v1/{table}?on_conflict=id"
        batch_size = 200
        all_ok = True

        for i in range(0, len(sanitized_rows), batch_size):
            chunk = sanitized_rows[i:i + batch_size]
            try:
                resp = requests.post(
                    endpoint,
                    headers=self._headers(key, {"Prefer": "resolution=merge-duplicates,return=minimal"}),
                    data=json.dumps(chunk),
                    timeout=20
                )
                if resp.status_code not in (200, 201, 204):
                    _warn_ui(f"Supabase BULK UPSERT `{table}` batch {i//batch_size + 1} failed (HTTP {resp.status_code}): {resp.text[:250]}")
                    all_ok = False
            except Exception as e:
                _warn_ui(f"Supabase BULK UPSERT `{table}` batch {i//batch_size + 1} exception: {e}")
                all_ok = False

        return all_ok


# Singleton store instance
_STORE = SupabasePostgRESTStore()


def get_storage_status() -> Dict[str, Any]:
    """Returns diagnostic metadata about active cloud vs. local fallback storage."""
    url, key = _get_supabase_credentials()
    return {
        "backend": "Supabase PostgreSQL (PostgREST)" if (url and key) else "Local JSON Fallback (Offline Mode)",
        "supabase_configured": bool(url and key),
        "users_table": USERS_TABLE,
        "records_table": RECORDS_TABLE
    }


# ==========================================
# PUBLIC BACKEND FUNCTIONS (PRESERVED SIGNATURES)
# ==========================================
def init_database() -> None:
    """
    Ensures default Admin ('admin' / 'admin123') and Demo Participant ('student1' / 'student123')
    exist in Supabase (or local JSON fallback) without overwriting existing records.
    """
    users = _STORE.fetch_all(USERS_TABLE, LOCAL_USERS_FILE)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    seeds_to_add: Dict[str, Dict[str, Any]] = {}

    if "admin" not in users:
        seeds_to_add["admin"] = {
            "username": "admin",
            "full_name": "Dr. Rohan Borgalli (Admin)",
            "department": "EXTC - SAKEC Chembur",
            "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
            "role": "admin",
            "created_at": now_str
        }
    if "student1" not in users:
        seeds_to_add["student1"] = {
            "username": "student1",
            "full_name": "Demo Participant (Trainee)",
            "department": "EXTC - SAKEC Chembur",
            "password_hash": hashlib.sha256("student123".encode()).hexdigest(),
            "role": "user",
            "created_at": now_str
        }

    if seeds_to_add:
        _STORE.upsert_many(USERS_TABLE, seeds_to_add, LOCAL_USERS_FILE)


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticates a user against Supabase `sakec_simlab_users` (or local JSON fallback)."""
    norm_u = _normalize_id(username)
    if not norm_u or not password:
        return None
    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    users = _STORE.fetch_all(USERS_TABLE, LOCAL_USERS_FILE)
    u_rec = users.get(norm_u)
    if u_rec and u_rec.get("password_hash") == pw_hash:
        return {
            "username": norm_u,
            "full_name": u_rec.get("full_name", norm_u),
            "department": u_rec.get("department", "EXTC - SAKEC"),
            "role": _normalize_role(u_rec.get("role", "user"))
        }
    return None


def register_new_user(username: str, full_name: str, department: str, password: str, role: str = "user") -> Tuple[bool, str]:
    """Registers a new user atomically in Supabase `sakec_simlab_users`."""
    norm_u = _normalize_id(username)
    if not norm_u or not full_name.strip() or not password:
        return False, "Please complete all required fields."

    users = _STORE.fetch_all(USERS_TABLE, LOCAL_USERS_FILE)
    if norm_u in users:
        return False, "Username already exists. Please choose a different username."

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_payload = {
        "username": norm_u,
        "full_name": full_name.strip(),
        "department": (department or "EXTC - SAKEC Chembur").strip(),
        "password_hash": hashlib.sha256(password.encode()).hexdigest(),
        "role": _normalize_role(role),
        "created_at": now_str
    }
    ok = _STORE.upsert_one(USERS_TABLE, norm_u, user_payload, LOCAL_USERS_FILE)
    if ok:
        return True, "Account registered permanently! Please sign in."
    return False, "Account saved locally, but cloud sync encountered an error (see warning banner)."


def save_module_progress(
    username: str,
    module_id: str,
    module_name: str,
    task_done: bool = False,
    quiz_score: int = 0,
    max_score: int = 3
) -> None:
    """
    Atomically upserts module task & quiz progress into `sakec_simlab_records`
    using composite key `<username>::<module_id>`, preserving the user's highest score.
    """
    norm_u = _normalize_id(username)
    norm_m = _normalize_id(module_id)
    composite_id = f"{norm_u}::{norm_m}"

    all_records = _STORE.fetch_all(RECORDS_TABLE, LOCAL_RECORDS_FILE)
    existing = all_records.get(composite_id, {})

    merged_task = bool(existing.get("task_completed", False)) or bool(task_done)
    merged_score = max(int(existing.get("quiz_score", 0)), int(quiz_score))
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    payload = {
        "username": norm_u,
        "module_id": norm_m,
        "module_name": str(module_name),
        "task_completed": merged_task,
        "quiz_score": merged_score,
        "max_score": int(max_score),
        "updated_at": now_str
    }
    _STORE.upsert_one(RECORDS_TABLE, composite_id, payload, LOCAL_RECORDS_FILE)


def fetch_user_progress(username: str) -> Dict[str, Dict[str, Any]]:
    """Returns a dictionary keyed by `module_id` for the specified participant."""
    norm_u = _normalize_id(username)
    all_records = _STORE.fetch_all(RECORDS_TABLE, LOCAL_RECORDS_FILE)
    user_map: Dict[str, Dict[str, Any]] = {}

    for rec_id, rec in all_records.items():
        if _normalize_id(rec.get("username", "")) == norm_u:
            m_id = str(rec.get("module_id", ""))
            if m_id:
                user_map[m_id] = {
                    "task_completed": bool(rec.get("task_completed", False)),
                    "quiz_score": int(rec.get("quiz_score", 0)),
                    "max_score": int(rec.get("max_score", 3)),
                    "updated_at": str(rec.get("updated_at", ""))
                }
    return user_map


def fetch_all_users_and_progress() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Fetches all users and all progress records for the Admin Analytics Dashboard."""
    users_map = _STORE.fetch_all(USERS_TABLE, LOCAL_USERS_FILE)
    records_map = _STORE.fetch_all(RECORDS_TABLE, LOCAL_RECORDS_FILE)

    users_list = []
    for uid, u in users_map.items():
        users_list.append({
            "username": u.get("username", uid),
            "full_name": u.get("full_name", uid),
            "department": u.get("department", "EXTC - SAKEC"),
            "role": _normalize_role(u.get("role", "user")),
            "created_at": u.get("created_at", "")
        })

    progress_list = []
    for rid, r in records_map.items():
        progress_list.append({
            "username": r.get("username", ""),
            "module_id": r.get("module_id", ""),
            "module_name": r.get("module_name", ""),
            "task_completed": bool(r.get("task_completed", False)),
            "quiz_score": int(r.get("quiz_score", 0)),
            "max_score": int(r.get("max_score", 3)),
            "updated_at": r.get("updated_at", "")
        })

    return users_list, progress_list
