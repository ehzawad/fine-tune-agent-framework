# Evaluation starter

This directory verifies API wiring and obvious tool-selection behavior. It is deliberately small and should not be confused with BFCL or tau-bench reproduction.

A deployment-grade evaluation should include:

- Hidden domain tasks and schema perturbations.
- Missing, ambiguous, conflicting, and malicious user instructions.
- Multi-turn state transitions and environment-grounded success checks.
- Read/write authorization, confirmation, and idempotency cases.
- Invalid enum, type, range, and extra-argument cases.
- Tool unavailability, timeouts, partial failures, and retries.
- Multiple deterministic trials plus sampled stress trials.
- Metrics for task success, tool name, argument validity, clarification, forbidden-action rate, false success claims, steps, latency, and cost.
