"""
Export the OmniVoice forward/logits subgraph to ONNX.

This script loads the pretrained OmniVoice model in FP16 on CUDA, wraps
the model so that only the logits tensor is returned, exports the wrapper
to ONNX, and validates the generated model structure.

The exported artifact represents only the forward/logits subgraph. It is
not the complete end-to-end OmniVoice text-to-speech pipeline.
"""

from pathlib import Path

import onnx
import torch
from omnivoice import OmniVoice


MODEL_ID = "k2-fsa/OmniVoice"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIRECTORY = PROJECT_ROOT / "models" / "onnx"
OUTPUT_FILE = OUTPUT_DIRECTORY / "omnivoice_forward.onnx"

DEVICE_NAME = "cuda:0"
ONNX_OPSET_VERSION = 18

BATCH_SIZE = 1
TOKEN_CHANNELS = 8
SEQUENCE_LENGTH = 8


class OmniVoiceLogitsWrapper(torch.nn.Module):
    """Wrap OmniVoice so ONNX export returns only the logits tensor."""

    def __init__(self, model: OmniVoice) -> None:
        """
        Initialize the export wrapper.

        Args:
            model: Loaded OmniVoice model.
        """
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the OmniVoice forward pass and return its logits.

        Args:
            input_ids: Input token tensor.
            audio_mask: Boolean mask identifying audio positions.
            attention_mask: Attention mask for valid sequence positions.
            position_ids: Positional indices for the input sequence.

        Returns:
            The logits tensor produced by OmniVoice.
        """
        output = self.model(
            input_ids=input_ids,
            audio_mask=audio_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        return output.logits


def create_example_inputs(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create representative tensors for ONNX tracing and validation.

    Args:
        device: CUDA device on which the tensors should be allocated.

    Returns:
        Input IDs, audio mask, attention mask, and position IDs.
    """
    input_ids = torch.zeros(
        (
            BATCH_SIZE,
            TOKEN_CHANNELS,
            SEQUENCE_LENGTH,
        ),
        dtype=torch.long,
        device=device,
    )

    audio_mask = torch.zeros(
        (
            BATCH_SIZE,
            SEQUENCE_LENGTH,
        ),
        dtype=torch.bool,
        device=device,
    )

    attention_mask = torch.ones(
        (
            BATCH_SIZE,
            SEQUENCE_LENGTH,
        ),
        dtype=torch.long,
        device=device,
    )

    position_ids = torch.arange(
        SEQUENCE_LENGTH,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    return (
        input_ids,
        audio_mask,
        attention_mask,
        position_ids,
    )


def validate_exported_model() -> None:
    """
    Confirm that the ONNX file exists and passes structural validation.

    Raises:
        FileNotFoundError: If the ONNX file was not generated.
        onnx.checker.ValidationError: If the model structure is invalid.
    """
    if not OUTPUT_FILE.exists():
        raise FileNotFoundError(
            f"ONNX file was not created: {OUTPUT_FILE}"
        )

    print("Checking ONNX structure and external weights...")

    onnx.checker.check_model(
        str(OUTPUT_FILE),
        full_check=False,
    )

    print("ONNX structural validation passed.")

    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)

    print(f"Main ONNX file size: {size_mb:.2f} MB")

    external_files = sorted(
        OUTPUT_DIRECTORY.glob(
            f"{OUTPUT_FILE.name}.data*"
        )
    )

    if external_files:
        print("External weight files:")

        for external_file in external_files:
            size_gb = external_file.stat().st_size / (1024**3)

            print(
                f"  {external_file.name}: "
                f"{size_gb:.2f} GB"
            )
    else:
        print("No external weight files were found.")


def main() -> None:
    """Load OmniVoice, export its logits wrapper, and validate the ONNX file."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. This export script requires an NVIDIA GPU."
        )

    device = torch.device(DEVICE_NAME)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading OmniVoice...")

    model = OmniVoice.from_pretrained(
        MODEL_ID,
        device_map=DEVICE_NAME,
        dtype=torch.float16,
        load_asr=False,
    )

    model.eval()

    wrapper = OmniVoiceLogitsWrapper(model)
    wrapper.eval()

    example_inputs = create_example_inputs(device)

    print("Testing wrapper...")

    with torch.inference_mode():
        expected_logits = wrapper(*example_inputs)

    print(
        "Wrapper output:",
        tuple(expected_logits.shape),
        expected_logits.dtype,
    )

    print(f"Exporting to: {OUTPUT_FILE}")

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            example_inputs,
            str(OUTPUT_FILE),
            input_names=[
                "input_ids",
                "audio_mask",
                "attention_mask",
                "position_ids",
            ],
            output_names=["logits"],
            opset_version=ONNX_OPSET_VERSION,
            dynamo=True,
            external_data=True,
            optimize=False,
            verify=False,
        )

    print("ONNX export command completed.")

    validate_exported_model()

    print()
    print("Export stage completed successfully.")


if __name__ == "__main__":
    main()