# Models

This directory contains model files used for inference and optimization.

## Directory Structure

```text
models/
├── onnx/
│   └── omnivoice_forward.onnx
│
├── tensorrt/
│   └── cache/
│
├── README.md
└── .gitkeep
```

---

# Included Files

The repository includes:

- `omnivoice_forward.onnx`

This ONNX model is used for ONNX Runtime and TensorRT inference.

---

# Excluded Files

Some generated artifacts are intentionally excluded from version control because they are automatically recreated and significantly increase repository size.

Examples include:

- `*.onnx.data`
- TensorRT engine files (`*.engine`)
- TensorRT cache files
- Timing cache files

These files are ignored through `.gitignore`.

---

# Regenerating Model Artifacts

Generate the ONNX model:

```bash
python scripts/export_onnx.py
```

Run ONNX Runtime benchmark:

```bash
python scripts/test_onnx_runtime.py
```

Run TensorRT benchmark:

```bash
python scripts/test_tensorrt_runtime.py
```

Running these scripts automatically recreates the required TensorRT engine and cache files.

---

# Notes

- TensorRT engine files are hardware-specific.
- Generated engine files should be rebuilt when using a different GPU architecture.
- Generated cache files are not required to run the repository and should not be committed.