const textInput =
  document.getElementById("textInput");

const instructionInput =
  document.getElementById("instructionInput");

const stepsInput =
  document.getElementById("stepsInput");

const stepsValue =
  document.getElementById("stepsValue");

const generateButton =
  document.getElementById("generateButton");

const statusElement =
  document.getElementById("status");

const audioPlayer =
  document.getElementById("audioPlayer");


stepsInput.addEventListener("input", () => {
  stepsValue.textContent = stepsInput.value;
});


function setMetric(id, value) {
  const element = document.getElementById(id);

  if (element) {
    element.textContent = value;
  }
}


function resetTelemetry() {
  setMetric("cpuUsage", "—");
  setMetric("processMemory", "—");
  setMetric("systemMemory", "—");
  setMetric("gpuUtilization", "—");
  setMetric("gpuMemory", "—");
  setMetric("gpuTemperature", "—");
}


generateButton.addEventListener("click", async () => {
  const text = textInput.value.trim();
  const instruction = instructionInput.value.trim();

  if (!text) {
    alert("Please enter text.");
    return;
  }

  generateButton.disabled = true;
  statusElement.textContent = "Generating speech...";

  resetTelemetry();

  try {
    const response = await fetch("/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text: text,
        language: "English",
        instruction: instruction,
        steps: Number(stepsInput.value),
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();

      throw new Error(
        `Speech generation failed: ${errorText}`
      );
    }

    const generationSeconds =
      response.headers.get("X-Generation-Seconds");

    const audioSeconds =
      response.headers.get("X-Audio-Seconds");

    const rtf =
      response.headers.get("X-RTF");

    const audioBlob = await response.blob();

    if (audioBlob.size === 0) {
      throw new Error(
        "The server returned an empty audio file."
      );
    }

    if (audioPlayer.src) {
      URL.revokeObjectURL(audioPlayer.src);
    }

    const audioUrl =
      URL.createObjectURL(audioBlob);

    audioPlayer.src = audioUrl;

    setMetric(
      "generationTime",
      generationSeconds
        ? `${generationSeconds} seconds`
        : "Unavailable"
    );

    setMetric(
      "audioDuration",
      audioSeconds
        ? `${audioSeconds} seconds`
        : "Unavailable"
    );

    setMetric(
      "rtf",
      rtf || "Unavailable"
    );

    const telemetryResponse =
      await fetch("/telemetry");

    if (!telemetryResponse.ok) {
      throw new Error(
        "Speech was generated, but telemetry could not be retrieved."
      );
    }

    const telemetry =
      await telemetryResponse.json();

    setMetric(
      "cpuUsage",
      telemetry.cpu_percent == null
        ? "Unavailable"
        : `${Number(
            telemetry.cpu_percent
          ).toFixed(1)}%`
    );

    setMetric(
      "processMemory",
      telemetry.process_memory_mb == null
        ? "Unavailable"
        : `${Number(
            telemetry.process_memory_mb
          ).toFixed(1)} MB`
    );

    setMetric(
      "systemMemory",
      telemetry.system_memory_percent == null
        ? "Unavailable"
        : `${Number(
            telemetry.system_memory_percent
          ).toFixed(1)}%`
    );

    setMetric(
      "gpuUtilization",
      telemetry.gpu_utilization_percent == null
        ? "Unavailable"
        : `${Number(
            telemetry.gpu_utilization_percent
          ).toFixed(1)}%`
    );

    setMetric(
      "gpuMemory",
      telemetry.gpu_memory_used_mb == null
        ? "Unavailable"
        : `${Number(
            telemetry.gpu_memory_used_mb
          ).toFixed(0)} MB`
    );

    setMetric(
      "gpuTemperature",
      telemetry.gpu_temperature_c == null
        ? "Unavailable"
        : `${Number(
            telemetry.gpu_temperature_c
          ).toFixed(0)} °C`
    );

    statusElement.textContent =
      "Generation complete.";

    try {
      await audioPlayer.play();
    } catch {
      statusElement.textContent =
        "Generation complete. Press Play to hear the audio.";
    }
  } catch (error) {
    console.error(error);

    statusElement.textContent =
      `Error: ${error.message}`;
  } finally {
    generateButton.disabled = false;
  }
});