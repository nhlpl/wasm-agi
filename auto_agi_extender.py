#!/usr/bin/env python3
"""
auto_agi_extender.py – Fully autonomous WASM extension generator and integrator.
No human intervention. Uses DeepSeek API to generate code, compiles, tests, and updates AGI.
"""

import os
import json
import time
import shutil
import subprocess
import tempfile
import sqlite3
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
import requests

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
TASKS_DIR = Path("./tasks")          # Watch this directory for new task JSON files
PROCESSED_DIR = Path("./tasks_processed")
FAILED_DIR = Path("./tasks_failed")
EXTENSIONS_DB = Path("./extensions.db")
AGI_LIB_RS = Path("./auto_lib.rs")   # Generated Rust source with WASM bytes
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "YOUR_API_KEY_HERE")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
WASMTIME = shutil.which("wasmtime")
CARGO = shutil.which("cargo")
RUST_TARGET = "wasm32-wasi"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AutoAGIExtender")

# ----------------------------------------------------------------------
# Database for tracking extensions
# ----------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(EXTENSIONS_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS extensions (
            name TEXT PRIMARY KEY,
            description TEXT,
            wasm_path TEXT,
            created_at REAL,
            test_cases TEXT
        )
    """)
    conn.commit()
    return conn

# ----------------------------------------------------------------------
# LLM client: generate Rust WASM template for a task
# ----------------------------------------------------------------------
def generate_rust_template(task: Dict) -> Optional[str]:
    """Call DeepSeek API to generate Rust code for the given task."""
    prompt = f"""You are an expert Rust programmer. Write a Rust function that solves the following problem and exports it as a WASM module with the signature `pub extern "C" fn solve(ptr: i32, len: i32) -> i32`.

The function should:
- Read input from linear memory at address `ptr` (length `len` bytes).
- Parse the input according to the specification.
- Compute the result.
- Write the result to linear memory at address 0x2000.
- Write the length of the result (in bytes) to address 0x2008.
- Return 0.

Task: {task['description']}
Input format: {task['input_type']}
Output format: {task['output_type']}

Provide only the Rust code, no explanation. Use `use std::slice;` and `unsafe` as needed.
"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 1500
    }
    try:
        resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        code = result['choices'][0]['message']['content']
        # Clean up code (remove markdown fences if present)
        if code.startswith("```rust"):
            code = code[7:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        return code.strip()
    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        return None

# ----------------------------------------------------------------------
# Compile Rust source to WASM
# ----------------------------------------------------------------------
def compile_to_wasm(source: str) -> Optional[bytes]:
    with tempfile.TemporaryDirectory() as tmpdir:
        src_dir = Path(tmpdir) / "src"
        src_dir.mkdir()
        src_path = src_dir / "lib.rs"
        src_path.write_text(source)
        cargo_toml = Path(tmpdir) / "Cargo.toml"
        cargo_toml.write_text(f"""
[package]
name = "temp_wasm"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]
""")
        # Run cargo build
        proc = subprocess.run(
            [CARGO, "build", "--target", RUST_TARGET, "--release"],
            cwd=tmpdir,
            capture_output=True,
            text=True
        )
        if proc.returncode != 0:
            logger.error(f"Compilation failed:\n{proc.stderr}")
            return None
        wasm_path = Path(tmpdir) / f"target/{RUST_TARGET}/release/temp_wasm.wasm"
        if not wasm_path.exists():
            logger.error("WASM file not found after compilation")
            return None
        return wasm_path.read_bytes()

# ----------------------------------------------------------------------
# Test WASM module with test cases
# ----------------------------------------------------------------------
def run_wasm(wasm_bytes: bytes, input_data: bytes) -> Optional[bytes]:
    with tempfile.NamedTemporaryFile(suffix=".wasm") as wasm_file:
        wasm_file.write(wasm_bytes)
        wasm_file.flush()
        proc = subprocess.run(
            [WASMTIME, wasm_file.name, "--invoke", "solve", "0", str(len(input_data))],
            input=input_data,
            capture_output=True
        )
        if proc.returncode != 0:
            logger.error(f"WASM execution failed: {proc.stderr.decode()}")
            return None
        return proc.stdout

def test_wasm(wasm_bytes: bytes, task: Dict) -> bool:
    test_cases = task['test_cases']
    for tc in test_cases:
        # Encode input based on input_type
        if task['input_type'] == "two integers":
            a, b = tc['input']
            inp = a.to_bytes(8, 'little') + b.to_bytes(8, 'little')
        elif task['input_type'] == "string":
            inp = tc['input'][0].encode()
        else:
            logger.error(f"Unknown input type: {task['input_type']}")
            return False
        out = run_wasm(wasm_bytes, inp)
        if out is None:
            return False
        # Decode output
        if task['output_type'] == "integer":
            result = int.from_bytes(out, 'little')
            expected = tc['expected']
        elif task['output_type'] == "boolean":
            result = out[0] != 0
            expected = tc['expected']
        else:
            logger.error(f"Unknown output type: {task['output_type']}")
            return False
        if result != expected:
            logger.error(f"Test failed: input={tc['input']}, expected={expected}, got={result}")
            return False
    return True

# ----------------------------------------------------------------------
# Store successful extension in database and update AGI library
# ----------------------------------------------------------------------
def store_extension(task: Dict, wasm_bytes: bytes):
    # Save WASM to a file
    wasm_dir = Path("./wasm_extensions")
    wasm_dir.mkdir(exist_ok=True)
    wasm_path = wasm_dir / f"{task['name']}.wasm"
    wasm_path.write_bytes(wasm_bytes)
    # Update database
    conn = sqlite3.connect(EXTENSIONS_DB)
    conn.execute("""
        INSERT OR REPLACE INTO extensions (name, description, wasm_path, created_at, test_cases)
        VALUES (?, ?, ?, ?, ?)
    """, (task['name'], task['description'], str(wasm_path), time.time(), json.dumps(task['test_cases'])))
    conn.commit()
    conn.close()
    # Regenerate the Rust source file that embeds all WASM extensions
    regenerate_rust_library()
    logger.info(f"Extension {task['name']} stored and library updated.")

def regenerate_rust_library():
    """Generate a Rust source file that contains static byte arrays for all WASM extensions."""
    conn = sqlite3.connect(EXTENSIONS_DB)
    cur = conn.execute("SELECT name, wasm_path FROM extensions")
    rows = cur.fetchall()
    conn.close()
    lines = ["// Auto-generated library from automatic extensions\n"]
    lines.append("use std::collections::HashMap;\n")
    lines.append("pub fn get_extension(name: &str) -> Option<&'static [u8]> {")
    lines.append("    let map: HashMap<&str, &[u8]> = HashMap::from([")
    for name, wasm_path in rows:
        # Read WASM bytes and embed as byte literal
        wasm_bytes = Path(wasm_path).read_bytes()
        # Convert to Rust byte array literal
        byte_str = ", ".join(str(b) for b in wasm_bytes)
        lines.append(f'        ("{name}", &[{byte_str}]),')
    lines.append("    ]);")
    lines.append("    map.get(name).copied()")
    lines.append("}")
    AGI_LIB_RS.write_text("\n".join(lines))
    logger.info(f"Regenerated {AGI_LIB_RS} with {len(rows)} extensions.")

# ----------------------------------------------------------------------
# Process a single task JSON file
# ----------------------------------------------------------------------
def process_task_file(task_path: Path):
    try:
        with open(task_path, 'r') as f:
            task = json.load(f)
    except Exception as e:
        logger.error(f"Failed to parse {task_path}: {e}")
        shutil.move(task_path, FAILED_DIR / task_path.name)
        return

    logger.info(f"Processing task: {task['name']}")
    # Attempt to generate code up to 3 times
    for attempt in range(3):
        source = generate_rust_template(task)
        if not source:
            logger.warning(f"LLM generation failed (attempt {attempt+1})")
            continue
        wasm_bytes = compile_to_wasm(source)
        if not wasm_bytes:
            logger.warning(f"Compilation failed (attempt {attempt+1})")
            continue
        if test_wasm(wasm_bytes, task):
            logger.info(f"Task {task['name']} succeeded!")
            store_extension(task, wasm_bytes)
            shutil.move(task_path, PROCESSED_DIR / task_path.name)
            return
        else:
            logger.warning(f"Tests failed (attempt {attempt+1})")
    # All attempts failed
    logger.error(f"Task {task['name']} failed after 3 attempts.")
    shutil.move(task_path, FAILED_DIR / task_path.name)

# ----------------------------------------------------------------------
# Directory watcher and main loop
# ----------------------------------------------------------------------
def watch_directory():
    # Create directories if not exist
    TASKS_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)
    FAILED_DIR.mkdir(exist_ok=True)
    init_db()
    # Regenerate initial library if needed
    if not AGI_LIB_RS.exists():
        regenerate_rust_library()
    while True:
        # Find all JSON files in TASKS_DIR
        for task_file in TASKS_DIR.glob("*.json"):
            process_task_file(task_file)
        time.sleep(10)  # Wait before scanning again

if __name__ == "__main__":
    # Ensure required tools are available
    if not WASMTIME:
        logger.error("wasmtime not found. Please install it.")
        sys.exit(1)
    if not CARGO:
        logger.error("cargo not found. Please install Rust.")
        sys.exit(1)
    # Check if wasm32-wasi target is installed
    proc = subprocess.run([CARGO, "target", "list"], capture_output=True, text=True)
    if "wasm32-wasi" not in proc.stdout:
        logger.info("Installing wasm32-wasi target...")
        subprocess.run([CARGO, "target", "add", "wasm32-wasi"], check=True)
    watch_directory()
