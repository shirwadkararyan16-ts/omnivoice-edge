from pathlib import Path

import onnx
import torch
from omnivoice import OmniVoice


MODEL_ID = "k2-fsa/OmniVoice"

OUTPUT_DIRECTORY = Path("models/onnx")
OUTPUT_FILE = OUTPUT_DIRECTORY / "omnivoice_forward.onnx"


class OmniVoiceLogitsWrapper(torch.nn.Module):
    """Expose only the logits tensor for ONNX export."""

    def __init__(self, model: OmniVoice) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        output = self.model(
            input_ids=input_ids,
            audio_mask=audio_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        return output.logits


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. This export script expects an NVIDIA GPU."
        )

    device = torch.device("cuda:0")

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading OmniVoice...")

    model = OmniVoice.from_pretrained(
        MODEL_ID,
        device_map="cuda:0",
        dtype=torch.float16,
        load_asr=False,
    )

    model.eval()

    wrapper = OmniVoiceLogitsWrapper(model)
    wrapper.eval()

    batch_size = 1
    token_channels = 8
    sequence_length = 8

    input_ids = torch.zeros(
        (
            batch_size,
            token_channels,
            sequence_length,
        ),
        dtype=torch.long,
        device=device,
    )

    audio_mask = torch.zeros(
        (
            batch_size,
            sequence_length,
        ),
        dtype=torch.bool,
        device=device,
    )

    attention_mask = torch.ones(
        (
            batch_size,
            sequence_length,
        ),
        dtype=torch.long,
        device=device,
    )

    position_ids = torch.arange(
        sequence_length,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    print("Testing wrapper...")

    with torch.inference_mode():
        expected_logits = wrapper(
            input_ids,
            audio_mask,
            attention_mask,
            position_ids,
        )

    print(
        "Wrapper output:",
        tuple(expected_logits.shape),
        expected_logits.dtype,
    )

    print(f"Exporting to: {OUTPUT_FILE}")

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (
                input_ids,
                audio_mask,
                attention_mask,
                position_ids,
            ),
            str(OUTPUT_FILE),
            input_names=[
                "input_ids",
                "audio_mask",
                "attention_mask",
                "position_ids",
            ],
            output_names=[
                "logits",
            ],
            opset_version=18,
            dynamo=True,
            external_data=True,
            optimize=False,
            verify=False,
        )

    print("ONNX export command completed.")

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

    print("ONNX structural validation passed.")
    print(
        f"Main ONNX file size: "
        f"{OUTPUT_FILE.stat().st_size / 1024 / 1024:.2f} MB"
    )

    external_files = list(
        OUTPUT_DIRECTORY.glob(
            f"{OUTPUT_FILE.name}.data*"
        )
    )

    if external_files:
        print("External weight files:")

        for external_file in external_files:
            size_gb = (
                external_file.stat().st_size
                / 1024
                / 1024
                / 1024
            )

            print(
                f"  {external_file.name}: "
                f"{size_gb:.2f} GB"
            )

    print()
    print("Export stage completed successfully.")


if __name__ == "__main__":
    main()