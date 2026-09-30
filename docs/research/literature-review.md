# Decision Models in Decision-Making: A Literature Review

**Date:** 2026-09-30
**Scope:** Decision models (Jev, OpenJev, Laya) vs. LLMs vs. humans in decision-making; hybrid architectures; robustness; organizational application.

---

## 1. The Decision Model Landscape

### 1.1 System One Decision Models

A new class of "System One" decision models has emerged in 2026. Unlike LLMs that generate text token-by-token, these models evaluate typed questions against a state and return structured decisions (choice, score, or probability) directly from model probabilities — no text generation, no parsing.

| Model | Vendor | Params | Latency | Calibration | Open? |
|-------|--------|--------|---------|-------------|-------|
| Jev | TypeSafe AI | undisclosed | 70–500ms | ECE 0.246 | Closed API ($0.042/1M input tokens) |
| Laya | Convai Innovations | 421M (ModernBERT-large) | 33ms | ECE 0.081 | Apache-2.0 |
| OpenJev | Community | 26B (DiffusionGemma) | ~50ms | — | MIT |

**Key primitives shared across models:**
- **Choice** — select from named options, returns probabilities + confidence
- **Score** — rate on an ordered rubric
- **Noul** — yes/no probability (0–1)

**Relevant papers:**
- Santos, J.A. (2026). *Calibrated Decision Models for Autonomous Penetration-Testing Harnesses: JEV and Laya as System One Decision Layers.* arXiv:2609.28940.
- TypeSafe AI documentation: https://docs.typesafe.ai/introduction
- Laya benchmarks: https://laya.convaiinnovations.com/

### 1.2 How They Differ from LLMs

| Dimension | Decision Models | LLMs |
|-----------|----------------|------|
| Output | Typed values (probabilities, choices) | Free text |
| Calibration | Designed for calibrated confidence | Often overconfident |
| Speed | 30–300ms | 500ms–30s |
| Explainability | Per-question confidence scores | Chain-of-thought (unreliable) |
| Failure mode | Can abstain / low confidence | Hallucination, sycophancy |
| Best use | Guardrails, routing, gates, scoring | Planning, reasoning, generation |

---

## 2. Decision Models vs. LLMs vs. Humans

### 2.1 Calibration & Overconfidence

**LLMs are systematically overconfident.** Humans are far better calibrated at comparable accuracy levels.

- **LLM overconfidence:** LLMs exhibit 2–4x more overconfidence than humans on QA tasks. Human overconfidence at baseline is ~4%; LLM exposure nearly triples human bias (Wang et al., 2025, arXiv:2505.02151).
- **Confidence patterns differ qualitatively:** Human confidence is bimodal with a mode at 50:50 ("I don't know"). LLM confidence is more uniformly distributed and less sensitive to task difficulty (Xu et al., 2025, ACL Findings).
- **Persona effects:** LLMs adopt stereotypically biased confidence when prompted with different personas (Asian persona more confident, female persona less confident) even when underlying accuracy is unchanged (Xu et al., 2025).

**Decision models (Jev/Laya) are explicitly designed for calibration.** Laya achieves ECE 0.081 vs. Jev's 0.246, both substantially better than typical LLMs.

### 2.2 Accuracy

- On typed-decisions benchmarks (2,000 decisions): Laya 0.766 vs. Jev 0.727 (Laya beats its 0.735 teacher ceiling).
- On AG News: Laya 0.950 vs. Jev 0.910.
- On emotion detection: Laya 0.595 vs. Jev 0.480.
- **No direct comparison of decision models vs. LLMs on identical decision tasks was found in the literature.** This is a gap.

### 2.3 Bias Amplification

- LLM input increases human accuracy by 5.6–7 percentage points but **more than doubles overconfidence** (Wang et al., 2025).
- Displaying LLM confidence further increases bias without improving accuracy.
- Humans with low baseline confidence benefit most from LLM input (+8.6–11.9pp accuracy) but also absorb the most bias.
- Humans with high baseline confidence are "immovable" — neither accuracy nor bias changes.

### 2.4 Emotional/State Effects

- Anxious priming of LLM agents reproduces human-like stress biases in consumer decision-making (npj AI, 2026). Across 2,250 runs, emotionally primed agents selected less healthy baskets, directionally consistent with human stress research.
- This suggests decision models may inherit or amplify state-dependent biases in ways that parallel human cognition.

---

## 3. Hybrid Architectures: Mixing Decision + Generative Models

### 3.1 Generative/Discriminative Hybrids (Classic)

- Raina et al. (2003, NeurIPS) showed hybrid generative/discriminative models outperform either alone, especially when training data is limited. Hybrid models produce better accuracy/coverage curves.
- This is the classical analogue of mixing decision (discriminative) and generative (LLM) models.

### 3.2 LLM Ensemble & Routing

- **AutoMix** (NeurIPS 2024): Routes between small and large LMs based on self-verification confidence. POMDP-based meta-verifier outperforms linear interpolation.
- **Hybrid-LLM** (ICLR 2024): Cost-efficient query routing between models.
- **MergeMix** (2025): Uses model merging weights as a proxy for optimal data mixing ratios.
- **LLM Ensemble survey** (arXiv:2502.18036): Comprehensive coverage of ensemble-before-inference (routing) and ensemble-after-inference (aggregation) methods.

### 3.3 System 1 / System 2 Alignment

- **Reasoning on a Spectrum** (arXiv:2502.12470): Explicitly aligns LLMs to System 1 (fast, intuitive) or System 2 (deliberate) thinking. Key finding: System 1-aligned models excel at commonsense; System 2-aligned at arithmetic/symbolic. A dynamic entropy-based combination outperforms both across nearly all benchmarks.
- **From System 1 to System 2 survey** (TPAMI 2026): Comprehensive review of reasoning LLMs.

### 3.4 Hybrid Human-AI Decision Structures

- **Shrestha et al. (2019, CMR):** Identifies three structural categories for organizational decision-making with AI:
  1. Full human-to-AI delegation
  2. Hybrid sequential (human-to-AI or AI-to-human)
  3. Aggregated human-AI decision making
- Contingency factors: specificity of search space, interpretability, size of alternative set, speed, replicability.

### 3.5 Does Mixing Help or Hurt?

**Evidence for mixing:**
- Hybrid generative/discriminative models beat either alone (Raina et al., 2003).
- Dynamic System 1/System 2 routing outperforms static approaches (arXiv:2502.12470).
- LLM ensembles with routing outperform single models (AutoMix, Hybrid-LLM).

**Evidence for harm:**
- Human-AI combinations on average perform **worse than the best of humans or AI alone** on decision tasks (Vaccaro et al., meta-analysis, 2024). Gains are significant only for content creation tasks.
- Mixing can amplify bias: LLM input doubles-to-triples human overconfidence (Wang et al., 2025).
- Lossy handoffs and correlated deliberation in multi-agent systems can underperform shared-state approaches (Liu, 2026, arXiv:2606.30986).

**Gap:** No studies found that specifically mix System One decision models with LLM generative nodes in a single decision pipeline and measure the effect on decision quality. This is a wide-open research direction.

---

## 4. Robustness in Decision Networks

### 4.1 Classical Robustness

- **Distributionally Robust Free Energy Principle** (Shafiei et al., 2025, arXiv:2503.13223): Wires robustness into agent decision-making by design. Agents complete tasks even when state-of-the-art models fail under distribution shift.
- **Adversarial robustness** for LLMs is well-studied but focuses on NER/classification, not decision pipelines.

### 4.2 Organizational Robustness

- **Resilience, robustness, antifragility** (Hillmann & Guenther, 2022): Reviews organizational resilience — the capacity to absorb disturbance and reorganize.
- **Agent Execution Control Layer** (2025): Centralized runtime governance for multi-agent systems improves robustness and accountability.
- **NIST AI RMF:** Framework for managing AI risk including robustness testing.

### 4.3 Do Human Organizational Methods Transfer to AI?

**Partial transfer:**
- Role clarity and specialization help both human and AI teams.
- Complementarity (assigning each party what they do best) applies.
- Shared-state architectures outperform human-imitation forms in multi-agent AI (Liu, 2026).

**Key differences:**
- AI agents lack motivation, identity, trust, socialization — coordination is sustained by "context architecture" (prompts, memory, schemas) rather than social mechanisms.
- Human organizational robustness relies on redundancy, slack, and social capital. AI robustness relies on validation, guardrails, and deterministic control layers.
- Lossy handoffs between AI agents create failure modes that don't occur in human teams (who can implicitly repair communication).

**Gap:** No research found that directly tests whether organizational robustness methods (e.g., redundancy, cross-training, rotation) produce the same results in AI decision networks as in human organizations.

---

## 5. Existing Human Decision-Making Datasets

### 5.1 Large-Scale Datasets

| Dataset | Domain | Size | Source |
|---------|--------|------|--------|
| choices13k | Risky choice under uncertainty | 240k+ judgments, 13k problems | Peterson et al. (2019), GitHub: jcpeterson/choices13k |
| CPC18 | Choice under risk/ambiguity/experience | 11k+ tasks | Erev et al. (2017), cpc-18.com |
| Strategic games | 2x2 matrix games | 93k decisions, 2,416 games | arXiv:2408.07865 |
| Psychological drivers | Economic decision-making meta-dataset | 1990–2025 literature | HuggingFace: juanmoisesdelas |
| Hotel game | Strategic decision-making | — | Shapira (2025), fine-tuned LLM generator |

### 5.2 Key Findings from Dataset Research

- **Dataset bias is significant:** Models trained on choices13k show systematic bias toward equipreference in stochastically dominated gambles, reflecting increased decision noise (Nat Hum Behav, 2024).
- **Cognitive model priors help:** Pretraining neural networks on synthetic data from prospect theory, then fine-tuning on real human data, achieves SOTA prediction of human decisions (Peterson et al., 2019).
- **BEAST-GB hybrid:** Integrating behavioral theory (BEAST) with gradient boosting captures >92% of predictable variation in human choice (Plonsky et al., 2025).
- **Fine-tuned LLM as data generator:** Fine-tuning an LLM on human data then using it to generate synthetic players achieves 80.1% accuracy on held-out humans (Shapira, 2025).

### 5.3 Relevance to Decision Models

These datasets are directly usable for:
- Benchmarking decision models (Jev, Laya) against human choice data
- Training cognitive model priors for decision models
- Studying whether decision models reproduce human-like biases
- Testing if hybrid architectures better predict human decisions

**Gap:** No published benchmarks of Jev/Laya/OpenJev on choices13k or CPC18 were found.

---

## 6. Research Gaps & Opportunities

### 6.1 Direct Comparisons Needed
1. **Decision models vs. LLMs on identical decision tasks** — no head-to-head benchmarks exist.
2. **Decision models vs. humans on the same information** — calibration, bias, and accuracy comparisons are missing.
3. **Hybrid decision+generative pipelines** — does adding an LLM reasoning step before/after a decision model improve or degrade outcomes?

### 6.2 Robustness Questions
4. Do organizational robustness methods (redundancy, cross-training) transfer to AI decision networks?
5. How do decision models behave under distribution shift compared to humans?
6. Does mixing decision nodes with generative nodes create compounding errors?

### 6.3 Dynamics & Networks
7. How do decision models behave in multi-agent / network settings vs. human teams?
8. Does the System 1/System 2 framing apply to decision model + LLM combinations?
9. Can decision models exhibit "organizational behavior" in the sense of Liu (2026)?

### 6.4 Data Opportunities
10. choices13k and CPC18 are immediately usable for benchmarking decision models against human data.
11. Fine-tuned LLM generators can create synthetic decision-making data for training.

---

## 7. Key References

### Decision Models
- Santos, J.A. (2026). Calibrated Decision Models... arXiv:2609.28940.
- TypeSafe AI. Jev documentation. https://docs.typesafe.ai/
- Laya AI. https://laya-ai.com/
- OpenJev. https://github.com/SiliconLabAI/OpenJev

### Human-AI Decision Making
- Wang et al. (2025). LLMs are overconfident and amplify human bias. arXiv:2505.02151.
- Xu et al. (2025). Do Language Models Mirror Human Confidence? ACL Findings.
- Spatharioti et al. (2025). Effects of LLM-based Search on Decision Making. CHI 2025.
- Vaccaro et al. (2024). When combinations of humans and AI are useful. Nature Human Behaviour.
- Shrestha et al. (2019). Organizational Decision-Making Structures in the Age of AI. California Management Review.

### Hybrid Architectures
- Raina et al. (2003). Classification with Hybrid Generative/Discriminative Models. NeurIPS.
- Reasoning on a Spectrum (2025). arXiv:2502.12470.
- AutoMix (2024). NeurIPS.
- Hybrid-LLM (2024). ICLR.
- LLM Ensemble survey (2025). arXiv:2502.18036.

### Robustness
- Shafiei et al. (2025). Distributionally Robust Free Energy Principle. arXiv:2503.13223.
- Liu, C. (2026). The Organizational Behavior of Agentic AI. arXiv:2606.30986.
- Hillmann & Guenther (2022). Resilience, robustness, and antifragility.

### Datasets
- Peterson et al. (2019). Cognitive Model Priors for Predicting Human Decisions. ICML.
- choices13k: https://github.com/jcpeterson/choices13k
- Plonsky et al. (2025). Predicting human decisions with behavioral theories and ML.
- Shapira (2025). Using LLMs to Simulate and Predict Human Decision-Making.

---

## 8. Suggested Next Steps

1. **Benchmark Jev/Laya on choices13k** — do decision models reproduce human choice patterns?
2. **Design a head-to-head study** — decision model vs. LLM vs. human on identical decision tasks.
3. **Build a hybrid pipeline** — decision model for routing/gating + LLM for reasoning, measure interaction effects.
4. **Test organizational robustness methods** — apply redundancy/cross-training to AI decision networks.
5. **Explore dynamical systems framing** — how do decision networks evolve over time with feedback?
