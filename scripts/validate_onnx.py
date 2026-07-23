"""
Validate the exported OmniVoice ONNX forward/logits model.

This script performs a structural validation of the exported ONNX model
using the official ONNX checker and prints the model's input and output
tensor shapes for quick inspection.

The validation targets the exported forward/logits subgraph rather than
the complete OmniVoice text-to-speech pipeline.
"""

from pathlib import Path

import onnx


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "onnx"
    / "omnivoice_forward.onnx"
)


def get_tensor_shape(value_info: onnx.ValueInfoProto) -> list[str | int]:
    """
    Extract a readable tensor shape from an ONNX value.

    Args:
        value_info: ONNX tensor metadata.

    Returns:
        A list containing static dimensions, symbolic dimensions,
        or '?' for unknown dimensions.
    """
    shape: list[str | int] = []

    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.dim_value:
            shape.append(dimension.dim_value)
        elif dimension.dim_param:
            shape.append(dimension.dim_param)
        else:
            shape.append("?")

    return shape


def main() -> None:
    """Validate the exported ONNX model and display its interface."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing ONNX model: {MODEL_PATH}"
        )

    print(f"Validating ONNX model:\n{MODEL_PATH}")

    onnx.checker.check_model(
        str(MODEL_PATH),
        full_check=False,
    )

    print("✓ ONNX structural validation passed.")

    model = onnx.load(
        str(MODEL_PATH),
        load_external_data=False,
    )

    print("\nModel inputs:")

    for model_input in model.graph.input:
        print(
            f"  {model_input.name}: "
            f"{get_tensor_shape(model_input)}"
        )

    print("\nModel outputs:")

    for model_output in model.graph.output:
        print(
            f"  {model_output.name}: "
            f"{get_tensor_shape(model_output)}"
        )


if __name__ == "__main__":
    main()