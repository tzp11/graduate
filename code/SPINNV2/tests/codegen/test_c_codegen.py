from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np

from tests.e2e.test_m1_e2e import _run_ort, _write_tiny_cnn


def test_generated_c_model_builds_runs_and_rejects_bad_checksum(tmp_path: Path):
    model_path = tmp_path / "tiny_cnn.onnx"
    spk_path = tmp_path / "tiny_cnn.spk"
    gen_dir = tmp_path / "generated"
    build_dir = tmp_path / "generated_build"
    input_path = tmp_path / "input.bin"
    output_path = tmp_path / "output.bin"

    _write_tiny_cnn(model_path)
    subprocess.run(
        [
            "python",
            "-m",
            "spinnv2.compiler",
            "compile",
            str(model_path),
            "-o",
            str(spk_path),
            "--target",
            "cpu_generic",
            "--external-inputs",
            "--external-outputs",
        ],
        check=True,
    )
    subprocess.run(
        [
            "python",
            "-m",
            "spinnv2.compiler",
            "codegen",
            str(spk_path),
            "--out-dir",
            str(gen_dir),
            "--name",
            "tiny",
            "--runtime-dir",
            "runtime",
        ],
        check=True,
    )
    assert (gen_dir / "tiny.c").exists()
    assert (gen_dir / "tiny.h").exists()
    assert "g_tiny_activation_arena" in (gen_dir / "tiny.c").read_text(encoding="utf-8")
    assert "spkv2_prepare_with_scratch" in (gen_dir / "tiny.c").read_text(encoding="utf-8")

    subprocess.run(["cmake", "-S", str(gen_dir), "-B", str(build_dir)], check=True)
    subprocess.run(["cmake", "--build", str(build_dir)], check=True)

    x = np.linspace(-1.0, 1.0, num=16, dtype=np.float32).reshape(1, 1, 4, 4)
    input_path.write_bytes(np.ascontiguousarray(x).tobytes())
    exe = _generated_executable(build_dir, "tiny_main_test")
    subprocess.run([str(exe), str(input_path), str(output_path)], check=True)

    actual = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    expected = _run_ort(model_path, x).reshape(-1)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    generated_spk = build_dir / "tiny.spk"
    assert generated_spk.exists()
    damaged = bytearray(generated_spk.read_bytes())
    damaged[len(damaged) // 2] ^= 0x01
    generated_spk.write_bytes(damaged)
    rejected = subprocess.run([str(exe), str(input_path), str(output_path)])
    assert rejected.returncode != 0


def _generated_executable(build_dir: Path, stem: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    candidates = [
        build_dir / f"{stem}{suffix}",
        build_dir / "Debug" / f"{stem}{suffix}",
        build_dir / "Release" / f"{stem}{suffix}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise AssertionError(f"generated executable not found: {candidates}")
