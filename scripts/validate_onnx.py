from pathlib import Path

import onnx


MODEL_PATH = Path(
    "models/onnx/omnivoice_forward.onnx"
)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing ONNX model: {MODEL_PATH}"
        )

    print(f"Validating: {MODEL_PATH}")

    onnx.checker.check_model(
        str(MODEL_PATH),
        full_check=False,
    )

    print("ONNX structural validation passed.")

    model = onnx.load(
        str(MODEL_PATH),
        load_external_data=False,
    )

    print()
    print("Model inputs:")

    for model_input in model.graph.input:
        dimensions = []

        for dimension in model_input.type.tensor_type.shape.dim:
            if dimension.dim_value:
                dimensions.append(dimension.dim_value)
            elif dimension.dim_param:
                dimensions.append(dimension.dim_param)
            else:
                dimensions.append("?")

        print(
            f"  {model_input.name}: {dimensions}"
        )

    print()
    print("Model outputs:")

    for model_output in model.graph.output:
        dimensions = []

        for dimension in model_output.type.tensor_type.shape.dim:
            if dimension.dim_value:
                dimensions.append(dimension.dim_value)
            elif dimension.dim_param:
                dimensions.append(dimension.dim_param)
            else:
                dimensions.append("?")

        print(
            f"  {model_output.name}: {dimensions}"
        )


if __name__ == "__main__":
    main()