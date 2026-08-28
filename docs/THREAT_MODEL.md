# Compact threat model

## Assets and trust boundaries

The protected assets are business state, customer data, credentials, tool permissions, audit integrity, and the operator's intent. The model server is a proposal boundary, not a trusted authorization boundary. User text, retrieved text, model output, tool arguments, and tool results are all untrusted until validated for their role.

## Principal failure paths

1. Prompt injection persuades the model to call a privileged tool.
2. The model fabricates a tool name, omits an argument, or supplies an unexpected field.
3. A valid-looking write exceeds the caller's permission or business limit.
4. A retry repeats a side effect.
5. Tool output contains malicious instructions that are fed back to the model.
6. Logs expose secrets or personally identifiable information.
7. A parser/runtime upgrade changes the tool-call contract.
8. A long-context request exhausts KV-cache capacity and degrades availability.

## Controls implemented here

- Known-name allowlist and strict Pydantic argument models with extra fields forbidden.
- Read/write risk classes and a deterministic policy decision before execution.
- Explicit approval for writes and a hard refund cap.
- Transactional SQLite state transitions and idempotent cancellation/refund behavior.
- Bounded loop length and duplicate-call protection.
- Structured error results and an append-only JSONL audit trail with basic redaction.
- A conservative raw-JSON fallback that accepts only an array of known calls.
- Pinned serving parameters and deterministic unit tests independent of model access.

## Controls deliberately left to a production system

Identity federation, RBAC/ABAC, secrets management, network isolation, connector-specific scopes, malware/content inspection, signed approvals, tamper-evident audit storage, data retention, rate limits, workload admission, disaster recovery, and continuous adversarial evaluation are outside this small reference implementation.
