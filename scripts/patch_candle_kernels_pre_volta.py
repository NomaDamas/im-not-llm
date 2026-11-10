"""Copies candle-kernels and patches it to build for pre-Volta GPUs (sm_6x).

candle's CUDA kernels assume compute capability >= 7.0 in two places:

* `reduce.cu` instantiates `sum_f16`, whose `atomicAdd(__half*, __half)`
  only exists from sm_70;
* the statically linked MoE kernels (`moe/moe_wmma*.cu`) use WMMA tensor-core
  fragments, also sm_70+.

On a Pascal card (TITAN Xp, GTX 10xx: sm_61) `nvcc` rejects both, so
`--features cuda` does not build. im-not-llm never calls either kernel (no
float16 `sum`, no mixture-of-experts layers), so this script compiles them out
for `__CUDA_ARCH__ < 700`; the MoE kernels trap if anything ever launches them.
float16 `sum` on such a card then fails with a missing-kernel error instead of
a wrong answer.

Run from the workspace root (Python 3, no packages):

    python scripts/patch_candle_kernels_pre_volta.py [OUT_DIR]

It writes `OUT_DIR.toml`, a cargo config that points cargo at the patched
copy, and prints how to use it:

    cargo build --release --features cuda --config OUT_DIR.toml
"""
import json
import pathlib
import shutil
import subprocess
import sys

WORKSPACE = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else WORKSPACE / "target" / "candle-kernels-pre-volta").resolve()

PRE_VOLTA = "defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700"
EDITS = {
    "src/reduce.cu": [
        ("SUM_OP(__half, sum_f16)\n", "#if !(" + PRE_VOLTA + ")\nSUM_OP(__half, sum_f16)\n#endif\n"),
    ],
    "src/moe/moe_wmma.cu": [
        ("using namespace nvcuda::wmma;\n", "#if !(" + PRE_VOLTA + ")\nusing namespace nvcuda::wmma;\n#endif\n"),
    ],
    "src/moe/moe_wmma_gguf.cu": [
        ("using namespace nvcuda::wmma;\n", "#if !(" + PRE_VOLTA + ")\nusing namespace nvcuda::wmma;\n#endif\n"),
        ("    using namespace nvcuda;\n", ""),
    ],
}
# Kernels whose bodies are WMMA: the body is replaced by a trap below sm_70.
KERNELS = {
    "src/moe/moe_wmma.cu": "__global__ void moe_gemm_grouped_kernel(",
    "src/moe/moe_wmma_gguf.cu": "__global__ void moe_gemm_gguf_prefill_kernel(",
}
TRAP = "#if " + PRE_VOLTA + "\n    __trap(); // WMMA needs sm_70 (patched by im-not-llm for pre-Volta builds)\n#else\n"


def kernels_package():
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--format-version", "1", "--features", "cuda"], cwd=WORKSPACE))
    for package in metadata["packages"]:
        if package["name"] == "candle-kernels":
            return package
    sys.exit("candle-kernels is not in the cuda dependency graph")


def guard_kernel_body(text, signature):
    """Wrap the body of the kernel declared by `signature` in `#if`/`#else`."""
    start = text.index(signature)
    open_brace = text.index(") {\n", start) + len(") {\n")
    depth, index = 1, open_brace
    while depth:
        char = text[index]
        depth += char == "{"
        depth -= char == "}"
        index += 1
    close_brace = index - 1
    return text[:open_brace] + TRAP + text[open_brace:close_brace] + "#endif\n" + text[close_brace:]


def main():
    package = kernels_package()
    source = pathlib.Path(package["manifest_path"]).parent
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(source, OUT)
    for name, edits in EDITS.items():
        path = OUT / name
        text = path.read_text(encoding="utf-8")
        for old, new in edits:
            if old not in text:
                sys.exit(f"{name}: expected text not found, candle-kernels {package['version']} changed: {old!r}")
            text = text.replace(old, new)
        if name in KERNELS:
            text = guard_kernel_body(text, KERNELS[name])
        path.write_text(text, encoding="utf-8", newline="\n")

    origin = package["source"] or ""
    if origin.startswith("git+"):
        table = 'patch."' + origin[len("git+"):].split("?")[0].split("#")[0] + '"'
    else:
        table = "patch.crates-io"
    config = OUT.with_suffix(".toml")
    config.write_text(f"[{table}]\ncandle-kernels = {{ path = '{OUT.as_posix()}' }}\n", encoding="utf-8")
    print(f"patched candle-kernels {package['version']} from {source} into {OUT}")
    print(f"build with: cargo build --features cuda --config {config}")


if __name__ == "__main__":
    main()
