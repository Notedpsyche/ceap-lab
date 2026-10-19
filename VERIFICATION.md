# Prototype verification

The prototype was checked on 20 October 2026 with its isolated Python 3.12 environment. The dependency versions are recorded in `uv.lock`; individual saved runs also include a SHA-256 digest of the simulator source.

## Automated checks

The test suite covers density-matrix physics, individual Bell branches, ideal chains of 2–5 nodes, noisy kernel/circuit agreement, Werner purification formulas, amplitude damping, memory composition, controller information limits, input privacy, illegal actions, memory reservations, explicit discards, resource exhaustion, routed classical delays, reproducibility, failure metrics, experiment grids, cancellation checkpoints, and API workflows.

The production **E0** experiment contains 16 analytical and independent Qiskit/Aer checks. The independent cases include all six ideal teleportation inputs, ideal purification, and noisy swapping/purification branches. All passed during verification.

The API test client emits an upstream Starlette/httpx deprecation warning; it does not affect the application's HTTP behavior.

## Browser acceptance

The Playwright workflow verifies:

- A valid editable initial configuration and rendered topology.
- Ideal automatic teleportation with fidelity 1.0000.
- A manual CEAP step, counterfactual evaluation, and manual action execution.
- Seeded comparisons with real results and confidence intervals.
- A prediction-calibration chart.
- Downloadable SVG, PNG, and CSV outputs.
- E0 execution through the production experiment worker.
- Cancellation retaining partial run state.
- Run history, stored-frame replay, and returning to live view.
- Desktop and 390-pixel mobile layouts, with no page-level horizontal overflow.
- No JavaScript page errors.

Screenshots are stored in `test-results/` (ignored generated artifacts), including `workbench-desktop.png`, `completed-desktop.png`, `counterfactual-desktop.png`, `comparison-desktop.png`, and `workbench-mobile.png`.

## Scope of the evidence

These checks establish working software and consistency with the specified small-chain model. Small browser-test batches are smoke tests, not research evidence that CEAP outperforms a baseline. A separate command-line comparison of immediate and fixed purification also completed and exported raw data. No hardware run or publication-novelty claim is made.
