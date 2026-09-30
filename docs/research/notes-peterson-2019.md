# Notes: Cognitive Model Priors for Predicting Human Decisions

**Paper:** Bourgin, Peterson, Reichman, Griffiths, Russell (2019). ICML 2019.
**arXiv:** 1905.09397

---

## Core Problem

Predicting human decisions under uncertainty is hard because:
1. Human behavior is noisy and variable across individuals
2. Behavioral datasets are small (hundreds to low thousands of problems)
3. Off-the-shelf ML overfits on small datasets
4. Theoretical models (prospect theory, BEAST) explain behavior but don't predict well at scale

## Key Contribution 1: Cognitive Model Priors

**Idea:** Use cognitive models to generate synthetic training data, then fine-tune on real human data.

**Method:**
1. Sample ~100k new problems from the space of possible CPC15/CPC18 problems
2. Use BEAST model to generate synthetic choice predictions for these problems
3. Train a neural network on (problem features → BEAST prediction)
4. Fine-tune the network on small real human datasets (learning rate 1e-6)

**Result:** The pretrained network outperforms all previous ML approaches and even the theoretical models themselves.

## Key Contribution 2: choices13k Dataset

- 242,879 human judgments on 13,006 problems (10x larger than CPC18)
- Collected via Amazon Mechanical Turk
- 16 participants per problem on average
- Feedback and no-feedback conditions
- Participants paid $0.75 + 10% bonus from random problem

## The BEAST Model

BEAST = Best Estimate and Sampling Tools

Predicts gamble value as: `V = BEV + ST + e`

Where:
- **BEV** = Best Estimate of expected value
- **ST** = Sampling Tools capturing 4 biases:
  1. Tendency to assume worst outcome
  2. Tendency to weight outcomes as equally likely
  3. Sensitivity to sign of reward
  4. Tendency to minimize immediate regret
- **e** = normal error term

## Experimental Setup (Human Interface)

From Figure 2 of the paper — the human prompt:

```
Please select option A or B.

Earning a Bonus. At the end of the experiment, one reward will be selected
at random from all the rewards you earned during the experiment. A fixed
proportion (10%) of this value will be paid to you as your performance
bonus for the task. If the sampled reward is negative, your bonus is set
to $0.00.
```

Then two gambles shown side-by-side. After choice, feedback shows:
- "In this trial, you chose B and gained 50"
- "Had you chosen A, you would have gained 16"

## Results Summary

### CPC15 Benchmark (MSE × 100)
| Model | MSE |
|-------|-----|
| ML + Raw Data (MLP) | 7.39 |
| ML + Raw Data (k-NN) | 7.15 |
| ML + Raw Data (Kernel SVM) | 5.52 |
| ML + Raw Data (Random Forest) | 6.13 |
| BEAST15 (theoretical) | 0.99 |
| CPC15 Winner | 0.88 |
| ML + Features (Random Forest) | 0.87 |
| ML + Features (Ensemble) | 0.70 |
| **MLP + Cognitive Prior (ours)** | **0.53** |

### CPC18 Benchmark (MSE × 100)
| Model | MSE |
|-------|-----|
| BEAST18 (theoretical) | 0.70 |
| ML + Features (Random Forest) | 0.68 |
| CPC18 Winner | 0.57 |
| **MLP + Cognitive Prior (ours)** | **0.48** |

### choices13k Benchmark (MSE × 100)
| Model | MSE |
|-------|-----|
| Linear Regression | 4.02 |
| k-Nearest Neighbors | 2.27 |
| Kernel SVM | 2.16 |
| Random Forest | 1.41 |
| MLP | 1.03 |
| Sparse MLP | 0.91 |

## Key Findings

1. **Cognitive priors help most when data is scarce:** At 4% of training data, prior reduces MSE from 0.020 to 0.0138
2. **Priors enable faster training:** Converges in fewer epochs
3. **Priors help even with large data:** Small but consistent advantage at 100% of choices13k
4. **ML + raw features alone fails:** Without theory-based features or priors, ML can't beat theoretical models
5. **Small test sets are unreliable:** CPC18-sized samples (210 problems) show high variance in model comparison; need ~6,500 problems for stable evaluation

## Relevance to Our Benchmark

### What we can learn:
- **Prompt matters:** Humans got explicit instructions about bonus structure — our models need the same
- **Raw features are hard:** ML models struggle with raw gamble parameters; cognitive structure helps
- **Dataset size:** choices13k is the right benchmark size; CPC18 is too small for stable conclusions
- **BEAST is the baseline to beat:** Any decision model should at least match BEAST's MSE

### What's different about our setup:
- We compare decision models (Jev, Laya) and LLMs directly against human data
- We don't train cognitive model priors — we test off-the-shelf models
- We measure calibration (ECE) in addition to accuracy/MSE
- We test whether LLMs can predict human decisions without explicit cognitive modeling

### Open Questions:
1. Do System One decision models (Jev/Laya) encode anything like BEAST's cognitive biases?
2. Can LLMs reproduce human-like deviations from expected utility?
3. Are decision models better calibrated than LLMs on human choice prediction?
4. Does the System 1/System 2 framing predict which model type will be better at predicting intuitive vs. deliberate human choices?
