# Model tags, digests and quantization

The models were served by Ollama on one NVIDIA L40 at temperature 0. The main perception
campaign (`perception_cache/phase_p.jsonl`) ran on 12-15 August 2026 and did not log model
digests. The table lists the tags installed on the experiment host when they were inspected
(November 2026). All tags were pulled again in late September 2026, after the main campaign
and before the corrected-prompt rerun and the later arms (`phase_p_v2*`, `phase_p_constrained`,
`phase_p_qwen3`, `phase_p_ref_constrained`, October 2026). The digests therefore identify the
models of the rerun and the later arms; for the main campaign a tag update cannot be excluded.

| Tag | Digest (first 12 hex) | Quantization | Last pulled |
| --- | --- | --- | --- |
| qwen2.5vl:7b | 5ced39dfa4ba | Q4_K_M | 2026-09-23 |
| gemma3:12b | f4031aab637d | Q4_K_M | 2026-09-24 |
| qwen2.5vl:3b | fb90415cde1e | Q4_K_M | 2026-09-30 |
| qwen2.5vl:32b | 3edc3a52fe98 | Q4_K_M | 2026-09-30 |
| llama3.2-vision:11b | 6f2f9757ae97 | Q4_K_M | 2026-09-30 |
| qwen3-vl:8b-instruct | 0533d74300e4 | Q4_K_M | 2026-10-02 |
