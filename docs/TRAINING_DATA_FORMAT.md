# Training data format

The trainer consumes UTF-8 JSONL. Each nonblank line is one trajectory object with `id`, `tools`, and `messages`.

## Minimal example

```json
{"id":"inventory-001","tools":[{"name":"check_inventory","description":"Check current inventory for one SKU.","parameters":{"type":"object","properties":{"sku":{"type":"string"}},"required":["sku"],"additionalProperties":false}}],"messages":[{"role":"user","content":"Check inventory for KB-75."},{"role":"assistant","tool_calls":[{"name":"check_inventory","arguments":{"sku":"KB-75"}}]},{"role":"tool","name":"check_inventory","content":{"status":"ok","data":{"sku":"KB-75","available_units":23}}},{"role":"assistant","content":"KB-75 has 23 units available."}]}
```

OpenAI-style wrappers are also accepted for tool declarations and calls:

```json
{
  "type": "function",
  "function": {
    "name": "check_inventory",
    "description": "Check current inventory for one SKU.",
    "parameters": {"type": "object", "properties": {}}
  }
}
```

## Allowed roles and sequence

A trajectory may start with one `system` message, followed by one or more cycles:

```text
user -> assistant text
user -> assistant tool call(s) -> tool result(s) -> assistant
```

After all results for parallel calls are present, the next message must be an assistant message. A system message is valid only at index zero.

## Tool definitions

Every trajectory carries the tools available in that scenario. Every tool requires:

- a unique nonempty `name`;
- a nonempty `description`;
- an object-shaped JSON Schema in `parameters`.

Use `additionalProperties: false` unless the tool truly accepts arbitrary keys. Enumerations, ranges, required fields, descriptions, and formats should reflect the real connector contract.

## Assistant tool calls

A call requires a declared tool name and object-shaped arguments. Arguments may be an object or a JSON-encoded object string. The trainer normalizes calls into OpenAI function form before rendering.

## Tool results

A tool result requires:

- `role: "tool"`;
- `name` matching one pending call;
- `content` containing the structured environment result.

Tool results should report what the environment actually did. Do not write a fabricated “success” observation merely because the desired call looked valid.

## Loss boundary

For every assistant turn, the trainer renders:

```text
prompt = all messages before this assistant turn + generation marker
completion = this assistant turn
```

Only completion tokens receive labels. Prompt tokens receive `-100` and do not contribute to cross-entropy loss. Each multi-turn trajectory therefore produces multiple supervised examples while preserving all prior context.

## Truncation policy

If prompt plus completion exceeds `max_seq_length`, the oldest prompt tokens are removed from the left. The completion is never silently truncated. A completion that cannot fit raises a validation error, because partially learning a JSON tool call is worse than rejecting the sample.

## Data-quality checks before training

At minimum, review:

1. tool schemas against the real APIs;
2. tool names and argument values against declared schemas;
3. user intent diversity, ambiguity, and missing-parameter cases;
4. correct tool observations and state transitions;
5. final answers that distinguish proposed from executed actions;
6. forbidden-action, denial, approval, timeout, and error trajectories;
7. duplicates, leakage between train and evaluation, and templated near-duplicates;
8. personally identifiable information and secrets;
9. license and provenance for every source;
10. held-out tasks that are not paraphrases of training examples.

The included demo rows prove the pipeline only. They are intentionally too small for meaningful adaptation.
