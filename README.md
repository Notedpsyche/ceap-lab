# CEAP Lab

A local research workbench implementing **Counterfactual Error-Aware Purification** from the team's `team report.pdf`. The browser is backed by a real Python density-matrix/event simulator. Charts are generated from recorded runs; there are no prefilled performance claims.

## Start

From this directory:

```bash
./run.sh
```

Open **http://127.0.0.1:8000**. Use `./run.sh --port 8080` to change ports. Stop the server with Ctrl+C. The existing project-local environment is ready to use.

To create a fresh environment on another machine, install [uv](https://docs.astral.sh/uv/) and run:

```bash
uv sync --python 3.12 --extra validation
./run.sh
```

`uv.lock` records the resolved dependencies. No Node.js, frontend build, cloud account, GPU, or hardware access is needed. Once installed, the application works offline; external links in Theory point to research sources.

## First walkthrough

1. Open **Workbench**, select **Ideal reference**, and choose **Immediate swapping**. Run the episode. All six standard inputs should reconstruct with fidelity one.
2. Choose **Uneven links** to restore the report's heterogeneous example. Run CEAP, or use **Step once** to watch resource generation and decisions.
3. Select **Evaluate actions** when paused. Each candidate has a predicted qualified-completion probability, expected extra time/attempts, and utility. Clicking a candidate executes it manually.
4. Click the network links or stored-pair arcs to inspect them. The displayed pair fidelity is an estimate available to the controller. Expand **Replay & diagnostics** only when you want simulator truth.
5. Use **Policy comparison** to run measured comparisons. The default is 200 episodes per policy; use 6 or 12 for a quick smoke test. A default noisy CEAP episode took about six seconds during development; runtime depends strongly on topology, memory, and rollout count.
6. **Experiments** provides E0–E7 from the team report. E0 runs analytical checks and independent Qiskit/Aer circuits. E4/E5/E7 have dedicated grids; other experiments accept the optional sweep.
7. **Run history** stores configurations, outcomes, events and versions. Inspect a run, replay an episode, or reproduce a saved configuration. Exports include trial CSV, manifest/config JSON, summaries, event/prediction JSONL, chart SVG/PNG, and a printable browser summary.

## What is implemented

- Fixed chains of 2–5 nodes and finite memory inventory.
- Werner, phase-flip, and exact amplitude-damping elementary resources.
- Independent seeded generation attempts, including failures and reserved memories.
- Parallel link preparation when endpoint capacity allows; reliable heralding with nonzero modeled delay.
- Local gate depolarization, recorded-bit errors, and per-qubit memory depolarization.
- Bell-measurement swapping and corrected endpoint teleportation.
- Two-copy BBPSSW-style purification, real accepted/rejected measurement branches, and an explicit ideal output twirl.
- Immediate, fixed-purification, threshold, weakest-link, CEAP, local ablation, and oracle-diagnostic policies.
- Counterfactual clones constructed from calibrated/observed state, separate from simulator truth.
- Server-sent progress; experiment workers in separate processes; cancellation checkpoints.
- SQLite run index and JSON/JSONL artifacts under `data/`.
- Responsive white interface with accessible controls and offline SVG charts.

## Model conventions

**Qubits:** the runtime uses leftmost-qubit-first tensors. Qiskit validation explicitly reverses little-endian ordering.

**Link fidelity:** the selected value is Bell overlap with Φ⁺. A Werner resource has equal populations in the other Bell states. Phase-flip resources mix Φ⁺ and Φ⁻. For amplitude damping on one half, `F = (1 + sqrt(1−γ))²/4`; calibration therefore covers F in [0.25, 1]. Equal overlaps do not mean identical channels.

**Gate error:** `Dλ(ρ) = (1−λ)ρ + λ(I/d ⊗ Tr_target ρ)` on the gate's own qubits. λ is not labeled average infidelity. The same specified λ is applied after each designated relay H or CNOT. Purification contains two local CNOTs; their noise channels act separately.

**Readout:** each Bell outcome bit flips independently with probability ε. The observer's conditional state sums over true outcomes consistent with the recorded bits. Simulator diagnostics can retain the hidden physical branch; policies cannot inspect it. Final Alice gates and receiver corrections are ideal, while final Bell readout uses ε.

**Memory:** each stored half has Bloch contraction `exp(−Δt/T)`. T=0 selects ideal memory. A newly generated pair ages through its acknowledgment delay. States are aged once per elapsed interval. To keep the event model small, quantum gates are placed at the operation-completion boundary after aging; this short-operation timing approximation is explicit.

**Purification:** two copies with the same endpoints are required. Both source pairs are removed. Accepted readout parity retains one state, computed from the actual branch, followed by an ideal free Werner twirl. Rejection retains none. A replacement generation action explicitly discards its named source pair before using that memory slot.

**Input:** the input is prepared once end-to-end entanglement is ready. The final receiver state integrates all Alice Bell/readout branches. This isolates unknown-input storage from resource preparation, while still including preparation in request latency. Six-state output diagnostics come from evaluating the same resource channel, not from copying the unknown input.

**Information limit:** the policy uses calibrated link fidelities, channel assumptions, pair ages, reported outcomes, budgets and availability. A fixed seeded calibration error is introduced by σ. Counterfactuals integrate over the public six-state input ensemble, never the actual unknown input label. The oracle uses true resource states but has the same planning algorithm; it is not a proven optimum.

## CEAP objective

For each legal action, sample 32 modeled futures by default. Execute the candidate, then follow the fixed-threshold continuation. Failed generation and rejected purification consume time/resources in the prediction.

```text
U(a) = P̂(qualified | a)
       − time_weight × E[extra milliseconds]/deadline
       − resource_weight × E[extra generation attempts]/attempt budget
```

Tie-breaking uses expected time, attempts, then action ID. Rollout streams are independent of execution randomness. Each candidate sees matched future seeds where events correspond. The output ensemble cycles through the six states; sample counts not divisible by six approximate equal weights.

Immediate swapping never purifies. Fixed purification seeks one accepted round per elementary link. Threshold seeks local estimated Bell fidelity 0.90. Weakest-link prioritizes lower estimated links. The local-quality ablation uses one-step purification gain without downstream rollouts. Pending prerequisites, memory limits and remaining budgets constrain every policy.

The prediction-calibration view is descriptive: multiple decisions within one episode have correlated final outcomes, and CEAP's later executed decisions can differ from the threshold continuation used in a forecast. The app states this distinction and does not attach unjustified independent-trial confidence intervals to those points.

## Metrics and reproducibility

- State fidelity: `⟨ψ|ρ_Bob|ψ⟩`, conditional on completed requests.
- Qualified probability: completed requests whose actual tested-input fidelity reaches the configured threshold, divided by all requests.
- Throughput: qualified completions divided by all sequential-request simulated duration, including failures.
- Time: generation, acknowledgment, gate/readout and routed correction delays; simulator compute time is separate.
- Resources: every generation attempt, successful elementary pairs, purification rejection, explicit discarded pairs, and peak per-node memory occupancy.
- Probability uncertainty: 95% Wilson intervals. Other reported intervals: 400 seeded bootstrap resamples. CEAP-vs-baseline qualified differences bootstrap matched scenario indices. Sparse completion cells are flagged.

Random draws are keyed by seed plus physical edge/attempt or operation/resource ancestry. Policies that create different events need not receive identical outcomes. Counterfactual evaluation never advances the execution stream.

`data/<run-id>/` contains the manifest, progress checkpoint, raw trials, events/predictions, or episode frames. The SQLite index is `data/lab.sqlite3`. The `CEAP_DATA_DIR` environment variable selects another local location. Reloading a page does not cancel a job. Restarting the server marks unfinished records interrupted; saved data remain inspectable. Cancellation is cooperative between completed trials, so a running CEAP episode finishes before the worker stops.

## Command-line reproduction

Run the same engine without the interface:

```bash
.venv/bin/python -m ceap experiment --episodes 6 --policies immediate,fixed,ceap --output data/my-comparison
```

Use the exported config or job request:

```bash
.venv/bin/python -m ceap experiment --config saved-config.json --episodes 200 --output data/reproduced-run
```

Use `--kind E4`, for example, for an experiment grid. Output directories must be empty. Runs save their exact configuration, version metadata, input-state sequence and seeds. Wall-clock policy runtime naturally varies between reproductions.

## Verification

```bash
.venv/bin/python -m pytest -q
```

With a local server running:

```bash
.venv/bin/python tests/browser_smoke.py
```

Browser tests use a locally installed Playwright Chromium or `CEAP_CHROMIUM=/path/to/chrome`. Install it on a new machine with `.venv/bin/python -m playwright install chromium`. Screenshots are saved under `test-results/`.

The suite verifies ideal branches, arbitrary supported chains, Werner purification success/output, noisy-operation kernels, local noise, memory composition, resource constraints, deadline handling, reproducibility, input/state privacy, API workflows, and independent Qiskit/Aer circuits. The independent check handles a known standalone-Python/Linux OpenMP TLS loading issue using a separate preloaded validation subprocess; it does not change runtime physics.

## Research status

This is a software prototype with documented model assumptions. It does not train PPO, claim hardware performance, guarantee that CEAP beats a heuristic, or establish publication novelty. Low-quality links and memory delay can make purification harmful; that is an outcome to measure.

The authoritative project direction is the supplied **team report.pdf**. Primary context: [Dür et al.](https://arxiv.org/abs/quant-ph/9808065), [QFOR](https://arxiv.org/abs/2508.04974), [Haldar et al.](https://doi.org/10.1103/PhysRevApplied.21.024041), and [Xiao et al.](https://www.nature.com/articles/s42005-024-01796-2).
