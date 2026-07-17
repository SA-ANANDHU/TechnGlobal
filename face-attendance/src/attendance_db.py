"""
attendance_db.py
-----------------
SQLite-backed storage for the attendance system: student roster and
attendance log, with a same-day duplicate guard (a student already marked
present today won't be logged twice).
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "attendance.db"
MODELS_DIR = ROOT / "models"


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist, and load the roster from label_map.json."""
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                roll_no TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                name TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                confidence REAL,
                FOREIGN KEY (student_id) REFERENCES students(student_id)
            )
        """)

        label_map_path = MODELS_DIR / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path) as f:
                label_map = json.load(f)
            for student_id, info in label_map.items():
                conn.execute(
                    "INSERT OR REPLACE INTO students (student_id, name, roll_no) VALUES (?, ?, ?)",
                    (student_id, info["name"], info["roll_no"]),
                )


def already_marked_today(student_id: str) -> bool:
    today = date.today().isoformat()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM attendance WHERE student_id = ? AND date = ? LIMIT 1",
            (student_id, today),
        ).fetchone()
        return row is not None


def mark_attendance(student_id: str, name: str, confidence: float) -> dict:
    """Insert an attendance record unless the student was already marked today."""
    if already_marked_today(student_id):
        return {"status": "already_marked", "student_id": student_id, "name": name}

    now = datetime.now()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO attendance (student_id, name, date, time, confidence) VALUES (?, ?, ?, ?, ?)",
            (student_id, name, now.date().isoformat(), now.time().strftime("%H:%M:%S"), confidence),
        )
    return {
        "status": "marked",
        "student_id": student_id,
        "name": name,
        "date": now.date().isoformat(),
        "time": now.time().strftime("%H:%M:%S"),
        "confidence": confidence,
    }


def get_attendance(date_filter: str = None, student_id: str = None) -> list:
    query = "SELECT * FROM attendance WHERE 1=1"
    params = []
    if date_filter:
        query += " AND date = ?"
        params.append(date_filter)
    if student_id:
        query += " AND student_id = ?"
        params.append(student_id)
    query += " ORDER BY date DESC, time DESC"

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def get_students() -> list:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM students ORDER BY name").fetchall()
        return [dict(r) for r in rows]


def get_attendance_summary(date_filter: str = None) -> dict:
    """Present/absent counts for a given date (defaults to today)."""
    date_filter = date_filter or date.today().isoformat()
    students = get_students()
    present_ids = {r["student_id"] for r in get_attendance(date_filter=date_filter)}
    return {
        "date": date_filter,
        "total_students": len(students),
        "present": len(present_ids),
        "absent": len(students) - len(present_ids),
        "present_ids": sorted(present_ids),
    }


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
    print(f"Students loaded: {len(get_students())}")
    result = mark_attendance("STU001", "Test Student", 0.97)
    print("Test mark:", result)
    print("Today's summary:", get_attendance_summary())
