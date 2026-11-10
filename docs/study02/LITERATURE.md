# Study 02 literature positioning

*Compiled 2026-11-10.*

## How far to trust this review

Every entry was checked against search-engine listings of its arXiv, proceedings or publisher page.
**No full text was read**: arxiv.org and aclanthology.org are outside this session's allowed network
domains, and the web-search budget ran out before the last checks. Claims are therefore at abstract
level. Entries marked ◦ were seen only in a secondary index or have an unconfirmed detail (venue,
author list or a headline number). Before citing any of this externally, read the five closest papers
(§2) in full and re-check every ◦ entry. arXiv IDs resolve at `https://arxiv.org/abs/<id>`.

## 1. What has been demonstrated

### Self-training on verified or filtered self-generated data

| work | shows | does not show |
|---|---|---|
| STaR, Zelikman et al., NeurIPS 2022 (2203.14465) | generate, keep answer-correct rationales, fine-tune, repeat | the authors report the loop cannot start when few-shot accuracy is near chance |
| ReST-EM, Singh et al., 2023 (2312.06585) | iterated sample, binary-correctness filter, fine-tune from base on MATH and APPS; beats human-data fine-tuning; gains grow with model size | small models, held-out task families |
| Haluptzok et al., ICLR 2023 (2207.14502) | self-generated programming puzzles verified by an interpreter more than double test accuracy | transfer beyond the puzzle format |
| Polu et al., ICLR 2023 (2202.01344) | verifier-gated expert iteration beats search alone at equal compute and orders its own difficulty curriculum | self-generated task diversity |
| Lee et al., ICML 2025 (2502.01612) | multi-round self-training extends addition from 10 to 100 digits; label noise causes an "error avalanche" | tool use, unseen task families |
| SEAL, Zweiger et al., 2025 (2506.10943) | a model learns to write its own fine-tuning data | cumulative curricula; forgetting under repeated edits ◦ |

### Self-proposed tasks and self-play with executable checks

| work | shows | does not show |
|---|---|---|
| Absolute Zero, Zhao et al., 2025 (2505.03335) | propose-and-solve code tasks validated by an executor; +5.7 to +13.2 points on 3B to 14B coder models | below 3B; tool use; gains shrink with size |
| Chae et al., 2025 workshop (2510.27072) | Absolute Zero is bounded by its base model and its proposer loses diversity | a remedy |
| Self-Challenging Agents, Zhou et al., 2025 (2506.01716) | an agent writes tool-use tasks with their own verification code, then trains on them; Llama-3.1-8B 12.0 → 23.5% | ≤2B, held-out families, a compute-matched comparison with distillation |
| **Tool-R0**, Acikgoz et al., Feb 2026 (2602.21320) | generator/solver self-play for tool calling from zero data; Qwen2.5-0.5B 15.5 → 30.6 and 1.5B 24.9 → 47.8 averaged over five public benchmarks | see §2 |
| **RLVE**, Zeng et al., ICML 2026 (2511.07317) | 400 difficulty-adaptive verifiable environments; a 1.5B model gains more from new environments than from 3× more training on its original data; more environments give better out-of-distribution results | tool use; environments are hand-built |
| R-Zero, Huang et al., 2025 (2508.05004) | challenger/solver co-evolution on a 4B model | executable verification ◦ |
| WebRL, Qi et al., ICLR 2025 (2411.02337) | a curriculum generated from failures takes an 8B web agent from 4.8 to 42.4% | judge-free verification (it uses a learned reward model) |
| Pu et al., NeurIPS 2026 (2605.22217) ◦ | a strict admission gate on proposed tasks is what stabilises self-play | beyond two toy tasks |

### Reinforcement learning with verifiable rewards

| work | shows | relevance |
|---|---|---|
| ToolRL, Qian et al., 2025 (2504.13958) | reward-decomposed GRPO beats SFT by 15% on 1.5B–7B models | policy-gradient RL works for tool calling at 1.5B; not tested at 0.5B or on self-generated tasks |
| Zhuang et al., CPAL 2026 (2506.13404) | RL on Qwen2.5-0.5B: GSM8K 45.5 → 54.0, MATH500 31.4 → 32.4 | gains at 0.5B are real but small |
| Yue et al., NeurIPS 2025 (2504.13837); Wu et al., 2025 (2507.14843); He et al., EMNLP 2025 (2506.02355) | verifiable-reward RL mostly sharpens solutions the base model can already sample; distillation from a teacher does add capability | the main reason a verified loop might not create *new* capability |
| DeepSeek-R1, 2025 (2501.12948) | reports that distilling into small students beats applying RL to them directly | cuts against the autonomous arm |
| Shenfeld et al., ICLR 2026 (2509.04259) | at matched new-task accuracy RL forgets less than SFT | retention across SFT cycles needs measuring |
| Shao et al., 2025 (2506.10947); Hochlehnert et al., 2025 (2504.07086) | random rewards produced large gains on one model family; many reported RL gains sit inside seed noise | motivates a no-verification control, several seeds and a second model family |

### Weakness-targeted data and curricula

| work | shows | does not show |
|---|---|---|
| LLM2LLM, Lee et al., ACL 2024 (2403.15042) | a teacher augments the student's errors each round | new task families; teacher dependence stays constant |
| DataEnvGym, Khan et al., ICLR 2025 (2410.06215); Montessori-Instruct, Li et al., ICLR 2025 (2410.14208) | teacher data generation driven by student feedback | adaptive vs uniform at matched compute on verifiable tasks |
| Matiisen et al., 2017 (1707.00183); Jiang et al., ICML 2021 (2010.03934); Parker-Holder et al., ICML 2022 (2203.01302) | learning-progress and regret-based curricula beat uniform sampling in game RL | LLM fine-tuning |
| Self-Evolving Curriculum, Chen et al., 2025 (2505.14970) | a bandit over problem categories beats random curricula for 3B–7B RL ◦ | unseen categories, SFT |
| **Wu et al., ICLR 2021 (2012.03107); Mordig et al., 2026 (2603.27226)** | negative results: curriculum ordering ≈ random except under short budgets or label noise; no robust gain on compositional deduction at a fixed budget | weakness-targeted task *selection*, which neither tests |

### Small-model tool use and unseen tools

| work | shows | relevance |
|---|---|---|
| Hammer, Lin et al., ICLR 2025 (2410.04587) | function-calling models overfit names; masking names with random strings forces models to read descriptions; variants down to 0.5B | precedent for Study 01's renaming result; single-turn only |
| ToolAlpaca, 2023 (2306.05301) | at equal instance count, more varied toolsets generalise better to unseen tools (10 → 400 tools ≈ 51 → 70% ◦) | the most direct "diversity, not volume" result for tools; 7B+, judged by an LLM |
| APIGen, NeurIPS 2024 (2406.18518); ToolACE, ICLR 2025 (2409.00920); xLAM, NAACL 2025 (2409.03215) | large verified synthetic function-calling datasets; 1B models competitive on BFCL | single-turn; no held-out multi-step structure |
| NESTFUL, 2024 (2409.03797) | GPT-4o reaches 28% full-sequence match on nested API sequences | deep chains are hard for frontier models too |
| Kang et al., NeurIPS 2025 (2505.17612) | teacher agent trajectories distilled into Qwen2.5 0.5B–3B students | the nearest distillation setting; two tools only |

### Compositional and depth generalisation

| work | shows | relevance |
|---|---|---|
| Dziri et al., NeurIPS 2023 (2305.18654) | transformers reduce multi-step composition to matching sub-graphs seen in training | matches Study 01: 72–74% on seen graphs, 26–38% on novel ones |
| Bogin et al., EMNLP 2022 (2201.05899) | a test item is hard when it contains a local structure absent from training | gives the definition of the primary endpoint: held-out *edges* |
| Oren et al., EMNLP 2021 (2109.02575) | 5K structurally diverse synthetic examples beat 1M uniformly sampled ones | strongest quantitative "structures, not volume" point; semantic parsing |
| Wang et al., EMNLP 2022 (2204.07705); Zhang et al., 2024 (2402.10891) | held-out-task performance grows with the number of training tasks and saturates quickly in instances per task; generalisation appears past a diversity threshold | predicts the Study 02 dose-response |
| Wei et al., ICLR 2022 (2109.01652) | more task clusters help unseen clusters only at sufficient model scale | capacity may bind at 0.5B |
| Anil et al., NeurIPS 2022 (2207.04901) | fine-tuning fails to length-generalise regardless of scale | ~0% at depth 5 is the expected outcome |
| Ramesh et al., ICML 2024 (2311.12997) | few training compositions generalise when intermediate outputs are produced step by step | the multi-turn runtime already externalises intermediate state |

### Continual learning and modular adapters

| work | shows | relevance |
|---|---|---|
| Biderman et al., TMLR 2024 (2405.09673) | LoRA learns less than full fine-tuning and forgets less | LoRA is the conservative choice for cycles |
| Shuttleworth et al., NeurIPS 2025 (2410.21228) | low-rank LoRA adds "intruder dimensions" that accumulate under sequential fine-tuning | a reason to prefer retraining from base over stacking updates |
| Scialom et al., EMNLP 2022 (2205.12393) | 1% rehearsal preserves earlier tasks over 8 sequential tasks | small replay fractions can suffice; full fine-tuning of a 3B model |
| Gerstgrasser et al., 2024 (2404.01413); Feng et al., ICLR 2025 (2406.07515) | accumulating data avoids model collapse; verification, even imperfect, prevents it | accumulate, never replace; verify everything |
| Lopez-Paz & Ranzato, NeurIPS 2017 (1706.08840) | accuracy matrix R[i, j], backward and forward transfer | the retention metrics used in Study 02 |
| Ostapenko et al., ICML 2024 (2405.11157); Huang et al., 2023 (2307.13269) | routing or weighting a library of LoRAs helps on held-out single tasks | modular reuse is feasible |
| Chen et al., 2025 (2506.13479); Tam et al., 2024 (2409.18314); Liu et al., ICML 2026 (2602.12323) ◦ | merging or routing adapters rarely matches a jointly trained adapter on real compositions, unless the composition was already in the training data | specialists are defensible for modular updates and retention, not for expected composition gains |
| Prabhakar et al., 2024 (2410.13025) | learned concatenation of LoRAs (CAT) composes two skills | needs composition examples; two skills only |

## 2. Closest prior work and what is missing

No work was found that shows a ≤2B model closing a held-out task-structure gap in multi-step tool use
through an autonomous verified loop, compared against static distillation at matched compute.

| closest work | what it has | what is missing for our question |
|---|---|---|
| Tool-R0 | 0.5B and 1.5B, no teacher, several self-play iterations, tool calling | evaluated on public benchmarks, not controlled held-out structures; no matched distillation arm; no retention measurement; independence of its verifier unconfirmed ◦ |
| Self-Challenging Agents | executable verification, a distillation setting | 8B; no held-out-family split |
| RLVE | 1.5B, adaptive verifiable environments, a compute-aware baseline, out-of-distribution gains from more environments | reasoning tasks on hand-built environments, not tool orchestration |
| Lee et al. 2025 | self-improvement that extends past the training range | synthetic arithmetic; no tool use |
| Oren et al. 2021; ToolAlpaca | diversity beats volume | not a dose-response over plan structures in a small tool-using model |

Three later preprints surfaced only through secondary indexes and should be read before any novelty
claim: SPADE (2608.19197), DART-SD (2608.18524) and 2606.23112 ◦.

## 3. The defensible contribution

The loop itself is not new. Three things are open, and Study 02 is designed around them:

1. **A dose-response for structure diversity.** No one has fixed the number of episodes and varied the
   number of distinct plan structures for a small tool-using model, with held-out structures as the
   outcome.
2. **A matched decomposition of the signal.** Published loops are compared with the base model or with
   baselines on different budgets. Study 02 compares teacher demonstrations, oracle trajectories and
   verified self-practice on the same task space, the same number of training episodes and the same
   optimizer steps, with rollout compute and teacher tokens reported as costs.
3. **Sealed families and family-level uncertainty.** Held-out families are fixed before any training,
   invisible to the curriculum, and resampled in the statistics, with retention measured each cycle.

A null result is informative on each: diversity not helping at 0.5B points to capacity, self-practice
falling short of demonstrations prices autonomy, and a curriculum that does not beat uniform sampling
would agree with Wu et al. and Mordig et al.

## 4. Why the loop might fail, according to this literature

| risk | source | how Study 02 addresses it |
|---|---|---|
| Filtering on verified success reinforces what the student can already do | Yue; Wu; He; STaR's near-chance caveat | a seed adapter, several samples per task, a curriculum aimed at tasks solved sometimes; base-versus-trained pass@k is reported |
| The task generator, not the learner, bounds transfer | RLVE; Tool-R0's frozen-generator ablation; Chae | Experiment 1 measures the ceiling of the task space with oracle labels before any loop is built |
| Capacity limits at 0.5B | Dziri; Wei; Li et al. 2025 (2502.12143) | a 1.5B arm in Experiment 1 and a pre-specified switch of student |
| Adaptive curricula add nothing at matched budget | Wu 2021; Mordig 2026 | uniform sampling is an arm; the curriculum hypothesis is stated as uncertain |
| Drift and forgetting across cycles | Shenfeld; Shuttleworth; Gerstgrasser | data accumulates and each cycle retrains from the base model; retention is measured every cycle |
| Gains that are not caused by the signal | Shao; Hochlehnert | a control that trains on unverified self-generated episodes; five seeds; a second model family |
| Composition by merging specialists fails | Chen; Tam; Liu | a jointly trained adapter is the reference in Experiment 3 |

## 5. Controls this literature treats as essential

Equal budgets that count rollouts, rejected samples and teacher tokens. Uniform-sampling and
static-data baselines. Correctly evaluated base and few-shot baselines. Several seeds with variance
reported. Held-out families frozen before the loop and invisible to generator, teacher and
curriculum. A control without the verification signal. More than one model family. Base-versus-trained
pass@k. Per-cycle retention. Teacher calls per cycle. All of these are in the Study 02 design
([PROPOSAL.md](PROPOSAL.md) §5–6).
