# Environment and Model Configuration

This file records the exact configuration used to produce the results reported in the
dissertation. LLM API providers periodically update or deprecate model versions, so exact
reproduction of the *generation* step (Section "3. Generate code from each Prompt condition"
in the README) is only possible while the same dated model snapshot remains available.

## Language model

- **Provider / access method:** OpenAI API (`chat.completions`), not the ChatGPT web
  interface. The web interface was used only during early, exploratory iteration on prompt
  wording (see `docs/known_issues.md` for why this distinction matters) and none of the
  data reported in the dissertation's quantitative results (Chapter 4) come from the web
  interface.
- **Model requested:** `gpt-4o-2024-08-06` (a dated snapshot, not the rolling `gpt-4o` alias,
  specifically to avoid the model silently changing between generation batches).
- **Model actually returned:** recorded per-call in `generation/generation_outputs/generation_log.csv`,
  column `model_returned`. This should match `model_requested` for every row; any mismatch
  indicates the requested snapshot was unavailable and should be investigated before trusting
  that batch's results.
- **Temperature:** `0.0`
- **Max tokens:** `4000` per response (see `finish_reason` in the log; any row with
  `finish_reason == "length"` was truncated and re-run)
- **Conversation history:** none. Each generation call is a single, independent request
  (`messages=[{"role": "user", "content": prompt_text}]`) with no prior turns, to eliminate
  cross-condition context leakage.
- **Repeats per condition:** 3 independent generations per Prompt condition.

## Python environment

- **Python version:** 3.13
- See `requirements.txt` for pinned package versions.

## Hardware

Calibration for the RAG condition (8 parameters, Differential Evolution,
`popsize=15`, `maxiter=1000`) is the most computationally expensive step in this repository.
Enabling `workers=-1, updating="deferred"` in `scipy.optimize.differential_evolution` is
recommended to parallelize across available CPU cores; no GPU is required anywhere in this
pipeline.

## Data window

- Calibration period: 1990-01-01 to 1999-12-31 (approx. 3,650 daily records, subject to
  missing-value gaps in the raw NDBC record)
- Validation period: 2000-01-01 to 2004-12-31
- Note: the calibration and validation periods in the processed data do **not** start on the
  same calendar day of year (see `docs/known_issues.md`, "Calendar/seasonal-phase alignment"),
  which is relevant to any ODE that includes a seasonal forcing term indexed by an internal
  solver clock rather than true day-of-year.
