# Models on build.nvidia.com, as seen from nvclaude

Measured with a real key from a residential connection. Re-measure with `nvclaude bench <model…>` or
`NVIDIA_API_KEY=… python3 -m unittest tests.test_live -v` (writes `docs/models.live.json`).

| model | first token | streams | tool calls | context | vision | notes |
|---|---|---|---|---|---|---|
| nvidia/nemotron-3-super-120b-a12b | ~1 s | yes | yes | 256K | no | default recommendation (`nvclaude super`) |
| nvidia/nemotron-3.5-lightning-30b-a3b | ~1 s | yes | yes | 1M | no | fastest; good for subagents (`nvclaude lightning`) |
| nvidia/nemotron-3-nano-omni-30b-a3b-reasoning | ~1 s | yes | yes | 1M | **yes** | answered "Blue" for a blue PNG (`nvclaude nano`) |
| nvidia/nemotron-3-ultra-550b-a55b | n/a | | | 1M | no | accepted requests; not yet benchmarked for latency (`nvclaude ultra`) |
| deepseek-ai/deepseek-v4-flash-0731 | ~6 s | yes | | 1M | no | high reasoning effort timed out at 120 s on a trivial prompt |
| deepseek-ai/deepseek-v4-pro-0813 | 187–255 s | **no** (whole response at once) | | 1M | no | NVIDIA's gateway returns 504 on full Claude Code requests; flagged SLOW |
| z-ai/glm-5.3-flash | | | | 1M | no | limit from API validation error |
| meta/llama-3.2-11b-vision-instruct | | | | 128K | yes | answered "Red" for a red PNG |
| nvidia/llama-3.1-nemotron-70b-instruct | | | | | | 404 "not found for account" — listed publicly but not usable on a free key |
| nvidia/llama-3.1-nemotron-ultra-253b-v1 | | | | | | 404 "not found for account" |
| nvidia/nemotron-nano-3-30b-a3b | | | | | | 404 "not found for account" |
| google/gemma-3-12b-it | | | | | | 404 "not found for account" |

Observations (2026-09-13/14):

- The public `/v1/models` list is not account-scoped: several ids 404 with "Function … not found for account". nvclaude records these and hides them.
- `stream_options.include_usage`, `reasoning_effort`, and `response_format` (json_schema) were accepted by Nemotron 3 Super.
- Text-only models reject images with `Received multimodal data but multimodal processing is not enabled`; nvclaude retries without the image.
- The free tier is rate limited (community reports ~40 requests/min) and can queue for minutes on large models; nvclaude keeps Claude Code's stream alive with pings while waiting.
