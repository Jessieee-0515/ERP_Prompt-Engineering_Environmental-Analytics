"""
Before running
1. pip install openai
2. Set your API key as an environment variable (see chat instructions),
   OR paste it directly into OPENAI_API_KEY below (less secure, fine
   for a personal one-off script).
3. Fill in PROMPTS below with your six actual prompt texts.
4. Check MODEL matches a real, currently available dated snapshot from
   https://platform.openai.com/docs/models
"""

import os
import csv
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from rag_retrieval import get_p6_rag_prompt, EXTENDED_TOP_K


#  Configuration -- edit this section

MODEL = "gpt-4o-2024-08-06"

TEMPERATURE = 0.0
N_REPEATS =  3         # how many independent generations per prompt
MAX_TOKENS = 4000

OUTPUT_DIR = Path("./generation_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SUMMARY_LOG = OUTPUT_DIR / "generation_log.csv"



PROMPTS = {
    "P1_Direct": """You are provided with two paired daily time series: Ta (air temperature, °C) and Tw (water temperature, °C), each as a 1D numpy array of equal length N, assumed already loaded as Ta_obs and Tw_obs. Generate a complete Python implementation of an Air2Water-style lake surface temperature model. Your implementation should: (1) define the governing ODE; (2) implement all model parameters; (3) numerically solve the ODE; (4) calibrate the model parameters against the provided Tw_obs data using any optimization method of your choice; (5) report a goodness-of-fit metric; (6) produce a visualization comparing simulated vs observed water temperature. Return: 1. Python code 2. A brief explanation of the implementation and your choice of calibration method.""",
   "P2_Role": """You are an environmental modelling expert specialising in hydrological models and parameter estimates based on physical processes.You are provided with two paired daily time series: Ta (air temperature, °C) and Tw (water temperature, °C), each as a 1D numpy array of equal length N, assumed already loaded as Ta_obs and Tw_obs. Generate a complete Python implementation of an Air2Water-style lake surface temperature model. Your implementation should: (1) define the governing ODE; (2) implement all model parameters; (3) numerically solve the ODE; (4) calibrate the model parameters against the provided Tw_obs data using any optimization method of your choice; (5) report a goodness-of-fit metric; (6) produce a visualization comparing simulated vs observed water temperature. Return: 1. Python code 2. A brief explanation of the implementation and your choice of calibration method""",
    "P3_Fewshot(simple)": """Before completing the following task, refer to these two examples of ODE-based modelling workflows.
    Example 1 — Soil Moisture Bucket Model
    dS/dt = P(t) - ET_max*(S/S_max) - k_drain*max(S - S_thresh, 0)
    S: soil moisture storage; P(t): precipitation input; ET_max: maximum
    evapotranspiration rate; S_max: maximum storage capacity; k_drain:
    drainage rate constant; S_thresh: threshold storage above which
    drainage occurs.
    Example 2 — Streeter–Phelps Dissolved Oxygen Model
    dL/dt = -k_d * L
    dD/dt = k_d * L - k_r * D
    L: biochemical oxygen demand (BOD); D: oxygen deficit; k_d: deoxygenation
    rate constant; k_r: reaeration rate constant.
    Use these examples only as demonstrations of the expected modelling
    workflow, programming structure, parameter calibration, and model
    evaluation. Do not reuse their equations, variables, assumptions, or
    parameter values. Instead, construct the Air2Water model strictly
    according to the task below.
    You are provided with two paired daily time series: Ta (air temperature,
    °C) and Tw (water temperature, °C), each as a 1D NumPy array of equal
    length N, assumed already loaded as Ta_obs and Tw_obs. Generate a
    complete Python implementation of an Air2Water-style lake surface
    temperature model. Your implementation should: (1) define the governing
    ODE; (2) implement all model parameters; (3) numerically solve the ODE;
    (4) calibrate the model parameters against the provided Tw_obs data
    using any optimization method of your choice; (5) report a
    goodness-of-fit metric; (6) produce a visualization comparing simulated
    vs observed water temperature. Return: Python code. A brief explanation
    of the implementation and your choice of calibration method.""",
    "P3_Fewshot(complex)": """Before completing the following task, refer to these two examples of ODE-based modelling workflows.Example 1 — Stream Temperature (logistic response)
    dTw/dt = (1/τ) * [Teq(Ta) - Tw]
    Teq(Ta) = μ + (α - μ) / (1 + exp(γ*(β - Ta)))
    Tw: stream temperature; Ta: air temperature; τ: response time (days).
    Teq follows an S-shaped curve, rising sharply only in a mid-range air
    temperature zone and flattening at high/low extremes.
    Example 2 — Ice-Cover Threshold Lake Model
    dTw/dt = K_open*(a + b*Ta - Tw)     if Tw > T_ice
    dTw/dt = K_ice*(0 - Tw)             if Tw <= T_ice
    Tw: lake temperature; Ta: air temperature; T_ice: ice-formation threshold (~0°C).
    K_ice << K_open — once ice forms, heat exchange with the atmosphere
    drops sharply and the lake relaxes toward 0°C much more slowly.
    Use these examples only as demonstrations of the expected modelling workflow, programming structure, parameter calibration, and model evaluation. Do not reuse their equations, variables, assumptions, or parameter values. Instead, construct the Air2Water model strictly according to the task below. You are provided with two paired daily time series: Ta (air temperature, °C) and Tw (water temperature, °C), each as a 1D NumPy array of equal length N, assumed already loaded as Ta_obs and Tw_obs. Generate a complete Python implementation of an Air2Water-style lake surface temperature model. Your implementation should: (1) define the governing ODE; (2) implement all model parameters; (3) numerically solve the ODE; (4) calibrate the model parameters against the provided Tw_obs data using any optimization method of your choice; (5) report a goodness-of-fit metric; (6) produce a visualization comparing simulated vs observed water temperature. Return: Python code. A brief explanation of the implementation and your choice of calibration method.""",
  "P4_CoT": """Before writing any code, reason step by step: (1) What physical relationship likely exists between air and water temperature (e.g., lag, damping, seasonal forcing)? (2) What mathematical ODE structure would capture this? (3) Given that the model will be calibrated against noisy real-world observations, what optimization strategy would be robust and appropriate (e.g., local vs global optimizer, avoiding overfitting)? Only after this reasoning, generate the implementationYou are provided with two paired daily time series: Ta (air temperature, °C) and Tw (water temperature, °C), each as a 1D numpy array of equal length N, assumed already loaded as Ta_obs and Tw_obs. Generate a complete Python implementation of an Air2Water-style lake surface temperature model. Your implementation should: (1) define the governing ODE; (2) implement all model parameters; (3) numerically solve the ODE; (4) calibrate the model parameters against the provided Tw_obs data using any optimization method of your choice; (5) report a goodness-of-fit metric; (6) produce a visualization comparing simulated vs observed water temperature. Return: 1. Python code 2. A brief explanation of the implementation and your choice of calibration method.""",
   "P5_Verification": """Write a complete Python implementation of an Air2Water-style lake surface
     temperature model: define the governing ODE, implement the parameters,
     solve it numerically, calibrate the parameters against observed data, and
     evaluate the fit.
     
     Before giving your final answer, review your own draft for these issues:
     - forcing data (e.g. air temperature) indexed/interpolated correctly by
       time, not broadcast incorrectly
     - any physical mechanism you describe (e.g. seasonal forcing) is actually
       implemented in the ODE, not dropped
     - any seasonal/time-dependent term uses the true calendar date, not an
       internal clock that resets to zero each run
     - the calibration method is suitable for a nonlinear, possibly multimodal
       objective (avoid a purely local optimizer without justification)
     - the ODE solver won't silently fail or diverge for extreme parameter
       values
     
     If you find any issues, you must fix them directly in the code, not just
     describe them. Give only the final corrected code, plus a short list of
     what you changed and why."""

}

# Build P6 (RAG) dynamically via hybrid retrieval:

P6_PROMPT_TEXT, P6_RETRIEVED_CHUNK_IDS = get_p6_rag_prompt(
    extended_top_k=EXTENDED_TOP_K
)
PROMPTS["P6_RAG"] = P6_PROMPT_TEXT

print(f"[P6_RAG] using {len(P6_RETRIEVED_CHUNK_IDS)} chunks "
      f"(core + top-{EXTENDED_TOP_K} extended): {P6_RETRIEVED_CHUNK_IDS}")


# 2. API client
client = OpenAI()

# 3. Generation function


def generate_once(prompt_text: str) -> dict:

    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "user", "content": prompt_text}
        ],
    )

    return {
        "text": response.choices[0].message.content,
        "model_returned": response.model,   # actual model that served the request
        "finish_reason": response.choices[0].finish_reason,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }

# 4. Batch loop over all prompt conditions and repeats

def run_batch():

    write_header = not SUMMARY_LOG.exists()

    with open(SUMMARY_LOG, "a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)

        if write_header:
            writer.writerow([
                "prompt_condition", "repeat_index", "timestamp_utc",
                "model_requested", "model_returned", "temperature",
                "finish_reason", "prompt_tokens", "completion_tokens",
                "output_filepath", "retrieved_chunks"
            ])

        for condition_name, prompt_text in PROMPTS.items():

            condition_dir = OUTPUT_DIR / condition_name
            condition_dir.mkdir(exist_ok=True)

            for i in range(1, N_REPEATS + 1):

                timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                print(f"[{condition_name}] generating repeat {i}/{N_REPEATS} ...")
                retrieved_str = (
                    ";".join(P6_RETRIEVED_CHUNK_IDS)
                    if condition_name == "P6_RAG" else ""
                )

                try:
                    result = generate_once(prompt_text)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    writer.writerow([
                        condition_name, i, timestamp,
                        MODEL, "ERROR", TEMPERATURE,
                        "error", "", "", "", retrieved_str
                    ])
                    continue

                out_path = condition_dir / f"{condition_name}_run{i}_{timestamp}.txt"
                out_path.write_text(result["text"], encoding="utf-8")

                writer.writerow([
                    condition_name, i, timestamp,
                    MODEL, result["model_returned"], TEMPERATURE,
                    result["finish_reason"],
                    result["prompt_tokens"], result["completion_tokens"],
                    str(out_path), retrieved_str
                ])
                log_file.flush()

                # Small delay to stay comfortably under rate limits.
                time.sleep(1)

    print(f"\nDone. Summary log saved to: {SUMMARY_LOG}")


if __name__ == "__main__":
    run_batch()
