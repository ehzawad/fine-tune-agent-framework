# Notice and model-license boundary

This repository contains an independently written training and agent-runtime implementation. It does not contain Salesforce model weights.

The model identifier `Salesforce/xLAM-2-32b-fc-r`, the xLAM name, and related marks belong to their respective owners. The upstream Hugging Face model card labels the checkpoint `CC-BY-NC-4.0` and describes it as a research release. Those terms are separate from this repository's Apache-2.0 license.

The file `templates/tool_chat_template_xlam_qwen.jinja` is vendored from the Apache-2.0-licensed vLLM repository, tag `v0.28.0`, path `examples/tool_chat_template_xlam_qwen.jinja`. It is included to keep training and serving serialization identical and reproducible. See `THIRD_PARTY_NOTICES.md`.

The demo trajectories and operations domain in this repository are independently authored examples. Salesforce datasets are referenced but not redistributed.

No historical benchmark value is an independent reproduction unless explicitly marked as such. The full model, A100 execution path, and online quality require verification on the target GPU host.
