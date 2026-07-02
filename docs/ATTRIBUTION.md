# Third-party attribution and licenses

This project's **own code** is licensed MIT (see [`../LICENSE`](../LICENSE),
© 2026 Zenity Ltd. / Z Labs). This platform loads, integrates with, or is trained
on a number of external models, datasets, and libraries — each governed by **its
own license**, several of which are more restrictive than MIT (gated Meta Llama
models, ShareAlike, non-commercial, and some with no stated license). The tables
below list what we use from each source, its license, and the obligation you must
honor. **You are responsible for accepting and complying with these upstream terms
when you download and use the models and data.** Licenses verified July 2026 — check
the linked source for the current terms before redistributing.

## Models

| Source | What we use | License | Obligation you must honor |
|---|---|---|---|
| [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Primary target model (activation capture) and `LlamaJudgeEvaluator` | **Llama 3.1 Community License** (custom; **gated**) | **Display "Built with Llama"**, retain the license notice, comply with the [Acceptable Use Policy](https://www.llama.com/llama3_1/use-policy/); use at/above 700M MAU needs separate Meta authorization |
| [meta-llama/Llama-Prompt-Guard-2-86M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M) | `PromptGuardEvaluator` (`pg_label`) — injection detection | **Llama 4 Community License** (custom; **gated**) | Same as above ("Built with Llama", license notice, AUP, 700M-MAU restriction) |
| [meta-llama/Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B) | `LlamaGuardEvaluator` (`lg_label`) — safety classification | **Llama 3.1 Community License** (custom; **gated**) | Same as above ("Built with Llama", license notice, AUP, 700M-MAU restriction) |
| [Goodfire/Llama-3.1-8B-Instruct-SAE-l19](https://huggingface.co/Goodfire/Llama-3.1-8B-Instruct-SAE-l19) (the `sae_lens` release `goodfire-llama-3.1-8b-instruct`, `layer_19`) | SAE feature extraction / interpretation / analysis | **Llama 3.1 Community License** (inherited; SAE trained on Llama-3.1 activations) | "Built with Llama", retain the license notice |
| Anthropic Claude via AWS Bedrock (`us.anthropic.claude-sonnet-4-20250514-v1:0`) | *Optional* LLM strategy in `SAEFeatureInterpreter` for SAE feature descriptions | Anthropic Commercial Terms + AWS Bedrock terms (paid API) | Comply with Anthropic's and AWS's terms of service |

**Built with Llama.** This project's core model and two of its four evaluators are
Meta Llama models, and the Goodfire SAEs and any classifier trained on Llama
activations (e.g. the bundled `demos/gradio_app/*.joblib`) are Llama derivatives.
Use is governed by the applicable **Llama Community License Agreement** (Llama 3.1
for the base model, Llama Guard 3, and the Goodfire SAE; Llama 4 for Prompt Guard 2)
and Meta's Acceptable Use Policy. The "Built with Llama" acknowledgment is displayed
here and in the [README](../README.md), and the license notices are retained. Use at
or above 700M monthly active users requires a separate license from Meta.

## Datasets

All datasets are loaded directly from Hugging Face (no dataset content is
redistributed in this repo). Each is governed by the license on its dataset card.
**Verify the current license and any use restrictions before redistributing data or
derived artifacts.** This repository uses them as a **research artifact** for the
paper's evaluation.

| Dataset | License (per HF card) | Gated? | Notes / obligation |
|---|---|---|---|
| [walledai/AdvBench](https://huggingface.co/datasets/walledai/AdvBench) | MIT | **Yes** | Mirror of AdvBench (Zou et al., [arXiv:2307.15043](https://arxiv.org/abs/2307.15043); orig. repo `llm-attacks/llm-attacks`, MIT). Retain attribution. |
| [walledai/HarmBench](https://huggingface.co/datasets/walledai/HarmBench) | MIT | **Yes** | Mirror of HarmBench (Mazeika et al., [arXiv:2402.04249](https://arxiv.org/abs/2402.04249); orig. repo `centerforaisafety/HarmBench`, MIT). Retain attribution. Includes a `copyright` subset. |
| [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | Apache-2.0 | No | Retain notice. |
| [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | **CC-BY-SA-3.0** | No | **ShareAlike copyleft** — attribution required; derivatives of the data must be shared alike (commercial use permitted). |
| [amanneo/enron-mail-corpus-mini](https://huggingface.co/datasets/amanneo/enron-mail-corpus-mini) | **No license stated** | No | Card says "More Information needed". Enron corpus data; treat as research-only and verify terms before redistribution. |
| [Lakera/gandalf_summarization](https://huggingface.co/datasets/Lakera/gandalf_summarization) | MIT | No | Retain attribution. |
| [jayavibhav/prompt-injection-safety](https://huggingface.co/datasets/jayavibhav/prompt-injection-safety) | **No license stated** | No | No license on card; treat as research-only and verify before redistribution. |
| [microsoft/llmail-inject-challenge](https://huggingface.co/datasets/microsoft/llmail-inject-challenge) | MIT | No | Retain attribution. |
| [Lakera/mosscap_prompt_injection](https://huggingface.co/datasets/Lakera/mosscap_prompt_injection) | MIT | No | Retain attribution. |
| [Open-Orca/OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) | MIT | No | Retain attribution. |
| [data-is-better-together/10k_prompts_ranked](https://huggingface.co/datasets/data-is-better-together/10k_prompts_ranked) | **`other` (unspecified)** | No | Card lists `license: other` with no further terms. Treat as research-only and verify before redistribution. |
| [qualifire/prompt-injections-benchmark](https://huggingface.co/datasets/qualifire/prompt-injections-benchmark) | **CC-BY-NC-4.0** | **Yes** | **Non-commercial only.** Attribution required; do not use for commercial purposes. |
| [xTRam1/safe-guard-prompt-injection](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection) | **No license stated** | No | No license field/tag; treat as research-only and verify before redistribution. |
| [SoftAge-AI/prompt-eng_dataset](https://huggingface.co/datasets/SoftAge-AI/prompt-eng_dataset) | MIT | **Yes** | Contact-info gate on HF. Retain attribution. |
| [allenai/wildjailbreak](https://huggingface.co/datasets/allenai/wildjailbreak) | **ODC-BY** | **Yes** | Attribution required; you must also accept the **AI2 Responsible Use Guidelines**. |
| [yanismiraoui/prompt_injections](https://huggingface.co/datasets/yanismiraoui/prompt_injections) | Apache-2.0 | No | Retain notice. |

## Libraries & external code

| Source | What we use | License | Obligation |
|---|---|---|---|
| [safety-research/circuit-tracer](https://github.com/safety-research/circuit-tracer) | *Optional* `circuit_tracer` backend — attribution-graph capture | MIT | Retain attribution |
| [microsoft/BIPIA](https://github.com/microsoft/BIPIA) | `BIPIADataset` imports BIPIA's attack builder from a **user-cloned** checkout (`bipia_root`); **not bundled here** | MIT (some bundled BIPIA *data* is CC-BY-SA-4.0 / MIT — see its LICENSE) | Clone and use under BIPIA's own license; cite Yi et al. |
| [uiuc-kang-lab/InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) | `InjecAgentDataset` reads InjecAgent data from a **user-cloned** checkout (`injecagent_root`); **not bundled here** | MIT | Clone and use under InjecAgent's own license; cite Zhan et al. |
| [Neuronpedia](https://www.neuronpedia.org) ([code](https://github.com/hijohnnylin/neuronpedia)) | `SAEFeatureInterpreter` / `NeuronpediaStrategy` fetch SAE auto-interpretations via the public API | Code: MIT. **Data/API output: no explicit license** | Attribute Neuronpedia; treat API data terms as unspecified |
| PyPI dependencies (see [`../pyproject.toml`](../pyproject.toml): `torch`, `transformers`, `transformer-lens`, `sae-lens`, `xgboost`, `zarr`, `polars`, `scikit-learn`, …) | Runtime dependencies | Their own OSI licenses (mostly MIT / BSD / Apache-2.0) | Retain their notices per each license |

## Gated resources (Hugging Face acceptance required)

Before the weights/data resolve, you must accept the license on the Hugging Face
page and authenticate with an `HF_TOKEN` (`huggingface-cli login`):

- **Models:** `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-Prompt-Guard-2-86M`,
  `meta-llama/Llama-Guard-3-8B`.
- **Datasets:** `walledai/AdvBench`, `walledai/HarmBench`,
  `qualifire/prompt-injections-benchmark`, `SoftAge-AI/prompt-eng_dataset`,
  `allenai/wildjailbreak`.

The Goodfire SAEs are **not** gated.

## Restrictive & unspecified licenses — read before commercial use or redistribution

- **Non-commercial:** `qualifire/prompt-injections-benchmark` is **CC-BY-NC-4.0** —
  do not use it for commercial purposes.
- **Copyleft / ShareAlike:** `databricks/databricks-dolly-15k` is **CC-BY-SA-3.0** —
  attribution and share-alike apply to the data and its derivatives.
- **Open Data with use guidelines:** `allenai/wildjailbreak` is **ODC-BY** and
  additionally requires accepting AI2's Responsible Use Guidelines.
- **No stated license (do not assume open — research use only, verify before
  redistributing):** `amanneo/enron-mail-corpus-mini`,
  `jayavibhav/prompt-injection-safety`, `xTRam1/safe-guard-prompt-injection`,
  `data-is-better-together/10k_prompts_ranked` (`other`, unspecified), and the
  Neuronpedia API/data output.
- **Custom Meta community licenses** (not OSI-approved; carry an Acceptable Use
  Policy and a 700M-MAU commercial restriction): all three Meta models and the
  Goodfire SAE, as listed under [Models](#models).
