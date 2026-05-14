# Linear Probing for Classical DSP Primitives Inside Black-Box AMC Models

**Complete Experimental Design**

---

## 1. Introduction & Motivation

Deep-learning-based Automatic Modulation Classification (AMC) models routinely outperform classical likelihood-based (LB) and feature-based (FB) detectors on standard benchmarks, yet generalize poorly under distribution shift (Snoap, Popescu, Latshaw & Spooner, *Sensors* 2023; Snoap & Spooner, *IEEE TBC* 2024). A central open question is whether such networks have implicitly rediscovered the cyclostationary and higher-order statistical primitives on which classical AMC is built, or whether they exploit spurious dataset-specific correlations.

This study tests the **Linear Representation Hypothesis** in the RF domain: that a high-capacity AMC network, if it relies on a quantity such as the cyclic autocorrelation magnitude at the symbol rate or the fourth-order cumulant $C_{40}$, will encode that quantity as a linearly decodable direction inside its hidden-state space. We further test whether **adversarial training** (Maroto, Bovet & Frossard, *EUSIPCO* 2022 — "SafeAMC") drives the model toward more *linearly aligned* representations of these physical primitives.

Two well-known pitfalls of probing (Hewitt & Liang, *EMNLP* 2019; Belinkov, *Comput. Linguist.* 2022) shape the design:
1. **Probe-class confounding.** DSP feature values are partially deterministic functions of the modulation class; high probe accuracy may therefore reflect class encoding rather than DSP-feature encoding. This is addressed via within-class variation datasets and control tasks.
2. **Correlation ≠ use.** A linearly decodable direction need not be causally used. This is addressed via LEACE-based amnesic ablation and steering interventions.

---

## 2. Theoretical Framework

### 2.1 Probing Equation
For a frozen AMC model $f$ with input $x \in \mathbb{C}^{N}$ (I/Q frame) and layer-$\ell$ activation $\phi_\ell(x) \in \mathbb{R}^{d_\ell}$, and a scalar DSP target $y_{\text{DSP}}(x) \in \mathbb{R}$ computed from $x$ by a known estimator, the linear probe is
$$
\hat{y}_{\text{DSP}} = w_\ell^\top \phi_\ell(x) + b_\ell, \quad w_\ell = \arg\min_{w} \sum_i (y_{\text{DSP}}(x_i) - w^\top \phi_\ell(x_i))^2 + \lambda \|w\|_2^2.
$$
The non-linear probe is a 2-layer MLP with the same input and target.

### 2.2 Validity Conditions
We declare that the model **explicitly represents** $y_{\text{DSP}}$ at layer $\ell$ only if **all three** of the following hold:

(V1) **Selectivity vs. control task** — probe $R^2$ on the true task exceeds probe $R^2$ on a random-label control by $\geq 0.3$ (Hewitt & Liang criterion).
(V2) **Within-class identifiability** — within at least one modulation class, varying only SNR / CFO / multipath while holding class fixed, $R^2 \geq 0.5$ above control-task baseline.
(V3) **Causal sufficiency** — applying LEACE erasure of the linear $y_{\text{DSP}}$ subspace at layer $\ell$ causes a *feature-specific* accuracy drop on modulations whose discrimination theoretically depends on $y_{\text{DSP}}$, while leaving other modulation pairs largely unaffected.

### 2.3 The "Linear Gap"
For each (layer, feature) pair, define
$$
\Delta_\ell(y) = R^2_{\text{MLP}}(\phi_\ell, y) - R^2_{\text{linear}}(\phi_\ell, y).
$$
$\Delta_\ell$ near zero indicates the feature is linearly aligned; large $\Delta_\ell$ indicates the feature is computable but not yet linearly extracted by the network at that depth. Crucially, the same gap must be computed against control-task baselines.

---

## 3. Target DSP Primitives

Primitives are organized in three tiers to manage estimator variance over 1024-sample frames.

### Tier A — Low-order statistics (high estimator confidence)
| Symbol | Definition | Estimator | Discriminates |
|---|---|---|---|
| $\sigma^2_{|x|}$ | Variance of instantaneous amplitude | sample variance of $|x[n]|$ | constant-envelope vs. amplitude-modulated |
| $\sigma^2_{f_i}$ | Variance of instantaneous frequency | sample variance of $\arg(x[n+1]x^*[n])$ | FSK vs. PSK / QAM |
| $K_{\text{spec}}$ | Spectral kurtosis | windowed periodogram kurtosis | bandwidth structure |
| $F_{\text{spec}}$ | Spectral flatness (Wiener entropy) | geometric/arithmetic mean of PSD | OFDM-like vs. single-carrier |

### Tier B — Second-order cyclostationary statistics
| Symbol | Definition | Estimator | Discriminates |
|---|---|---|---|
| $|R^\alpha_x(\tau)|$ at $\alpha=1/T$ | Cyclic autocorrelation magnitude at symbol rate | Time-Smoothing Method (TSM) with 16 sub-blocks, Hamming window | symbol-rate presence |
| $|S^\alpha_x(f)|$ at $\alpha=1/T, f=0$ | Spectral correlation function peak | FFT Accumulation Method (FAM), $N=64$, $P=128$ | linearly modulated digital signals |

### Tier C — Higher-order cumulants
| Symbol | Definition | Estimator | Discriminates |
|---|---|---|---|
| $C_{20}$ | $\mathbb{E}[x^2]$ | sample moment | PSK family (vanishes) vs. others |
| $C_{40}$ | $\mathbb{E}[x^4] - 3(\mathbb{E}[x^2])^2$ | unbiased 4th-order cumulant estimator | QPSK vs. 16-QAM vs. 64-QAM |
| $C_{42}$ | $\mathbb{E}[|x|^4] - |\mathbb{E}[x^2]|^2 - 2(\mathbb{E}[|x|^2])^2$ | unbiased 4th-order cumulant estimator | PSK vs. QAM family |

$C_{60}$ is excluded; at 1024-sample frame length and the SNR regimes considered, its estimator variance is too high to serve as a reliable probe target. It can be added in a follow-up study with longer frames.

For each feature, ground-truth values are computed once and cached. Each frame's $y_{\text{DSP}}$ is stored alongside the frame index. We additionally compute, per (modulation, SNR) bin, the **estimator standard deviation** $\hat{\sigma}_{y_{\text{DSP}}}$ from a held-out calibration set of 5000 i.i.d. realizations; probe $R^2$ above $1 - \hat{\sigma}^2_{y_{\text{DSP}}} / \text{Var}(y_{\text{DSP}})$ is interpreted as approaching estimator-noise ceiling.

---

## 4. Models & Training

### 4.1 Architectures
- **VT-CNN2** (O'Shea & West, *GNU Radio Conf.* 2016): 2 conv layers, 2 FC layers. Probe points: post-conv1, post-conv2, post-FC1 (3 points).
- **ResNet-33** (DeepSig benchmark, RML2018.01A reference): probe points at the output of each of the 6 residual stages and final pooling (7 points).
- **Adversarially-trained ResNet-33 (AT-ResNet)**: identical architecture, trained as in §4.3.

### 4.2 Standard Training
- Dataset: RadioML 2018.01A, all 24 classes, all SNR values $\geq -6$ dB.
- Optimizer: Adam, lr $10^{-3}$, cosine decay over 50 epochs.
- Batch size 256. Cross-entropy loss.
- Three independent seeds (12, 34, 56). All downstream analyses reported across seeds.

### 4.3 Adversarial Training (AT-ResNet)
- Threat model: $\ell_2$-bounded perturbation $\delta$ in the I/Q sample space with $\|\delta\|_2 / \|x\|_2 \leq 0.1$ (the SafeAMC budget).
- Inner attack: PGD with 7 steps, step size $\|\delta\|_2 / 4$.
- Otherwise identical to standard training.

### 4.4 Training Validity Check
Before any probing, confirm that all three models exceed published baselines:
- VT-CNN2: $\geq 56\%$ top-1 at SNR $\geq 0$ dB.
- ResNet-33: $\geq 92\%$ top-1 at SNR $\geq 10$ dB.
- AT-ResNet: clean accuracy within 3% of ResNet-33; PGD-perturbed accuracy $\geq 20$ pp higher than ResNet-33.

---

## 5. Datasets

### 5.1 Primary — RadioML 2018.01A (filtered)
- Filter to SNR $\geq 10$ dB to ensure DSP estimator stability over 1024-sample frames.
- Resulting set: ~640k frames across 24 modulations.
- Split: stratified by (class, SNR) but **partitioned by example index, not random shuffle**, into 70% train / 15% val / 15% test. This avoids leakage from temporally adjacent frames that share an underlying realization.

### 5.2 Within-Class Augmentation Set (WCA-Set) — **critical for V2**
This is the dataset that breaks probe-class confounding by introducing controlled DSP-feature variation *without* changing the modulation class.

Construction (GNU Radio + custom NumPy pipeline):
- For each of 8 representative modulations (BPSK, QPSK, 8-PSK, 16-QAM, 64-QAM, GFSK, OOK, 4-ASK):
  - **SNR sweep:** 10, 15, 20, 25, 30 dB.
  - **CFO sweep:** $\pm 0, \pm 200, \pm 400, \pm 800$ Hz at 200 kHz sample rate.
  - **Multipath sweep:** AWGN-only, 3GPP EPA-5Hz, EVA-70Hz, ETU-300Hz.
  - **Symbol-rate jitter:** $\pm 0.5\%$ at four levels.
- 8 classes × 5 SNR × 9 CFO × 4 multipath × 4 rate-jitter = 5760 (class, condition) cells × 100 frames each $= 576\text{k}$ frames.
- Each frame's true DSP values vary continuously within a class; this is the within-class variation required by V2.

### 5.3 Splits
- For WCA-Set, split by **generator seed**, not frame, so that no realization appears in both train and test of the probe.
- Three split seeds; results reported as mean $\pm$ std across seeds.

---

## 6. Probing Protocol

### 6.1 Activation Caching
1. Freeze model. Forward-pass the relevant split through the model.
2. Cache layer-$\ell$ activations as float16 tensors, flattened to 1D per frame.
3. Cache size estimate: ResNet-33 stage-5 activation is ~16k floats; over 600k frames this is ~20 GB per layer in float16. Store to NVMe SSD; load on demand.

### 6.2 Linear Probes
- **Method:** ridge regression with $\lambda$ selected by 5-fold CV on the probe training split.
- **Target normalization:** per-feature z-scoring on training set; same scaler applied to validation/test.
- **Metric:** test $R^2$ and Pearson $r$.

### 6.3 Non-linear Probes (Linear Gap)
- 2-layer MLP, hidden width 256, ReLU, dropout 0.1.
- Train with Adam, lr $10^{-3}$, early-stopping on val MSE with patience 10.
- Report $\Delta_\ell(y) = R^2_{\text{MLP}} - R^2_{\text{linear}}$.

### 6.4 Control-Task Baseline (V1)
- For every (model, layer, feature) probe, repeat the exact protocol with **permuted targets** (random permutation of the $y$ array across the training set, fixed seed).
- Report selectivity $S_\ell(y) = R^2_{\text{true}} - R^2_{\text{control}}$.
- A feature passes V1 if $S_\ell(y) \geq 0.3$ for at least one layer.

### 6.5 Within-Class Probing (V2)
- Use WCA-Set only.
- For each of the 8 within-class modulations, fit a separate probe; report per-class $R^2$ and the mean across the 8 within-class probes.
- A feature passes V2 if the mean within-class $R^2$ exceeds the within-class control-task baseline by $\geq 0.3$ at some layer.

### 6.6 SNR Stratification
- Repeat all probes restricted to each SNR slice $\{10, 15, 20, 25, 30\}$ dB.
- Plot probe $R^2$ vs. SNR for each feature and layer. Compare against the modulation-classification accuracy curve to see whether DSP-feature probe accuracy degrades faster or slower than classification accuracy as SNR drops.

---

## 7. Causal Interventions

### 7.1 LEACE Amnesic Ablation (V3)
- Library: `concept-erasure` (Belrose, Furman, Smith, Halawi, Ostrovsky, McKinney, Biderman, Steinhardt, *NeurIPS* 2023).
- For each layer $\ell$ and feature $y$, fit LEACE eraser $P_{\ell,y}$ on the probe training split.
- Forward-pass test data; at layer $\ell$, apply $\phi_\ell \mapsto P_{\ell,y}\phi_\ell$ and propagate.
- Report:
  - Overall classification accuracy drop $\Delta\text{Acc}$.
  - Per-modulation-pair accuracy drop on **dependent pairs** (theoretically discriminated by $y$) vs **independent pairs**.
  - Example dependence map:
    - $C_{40}$: dependent on (16-QAM, 64-QAM); (16-QAM, 256-QAM). Independent on (QPSK, 8-PSK); (BPSK, GFSK).
    - $|R^\alpha_x|$ at $\alpha = 1/T$: dependent on (digital-linear, FM/AM analog).
    - $\sigma^2_{f_i}$: dependent on (FSK, PSK).
- V3 passes if the dependent-pair drop is at least 3× the independent-pair drop.

### 7.2 Iterative Nullspace Projection (INLP)
- Run as cross-check on LEACE.
- Method: Ravfogel, Elazar, Goldberg, *ACL* 2020 — iteratively fit linear probes and project to their nullspace until probe $R^2 < 0.1$.
- Report agreement of accuracy drops within $\pm 2$ pp of LEACE.

### 7.3 Steering Vectors
- For probe direction $w_\ell^{(y)}$, normalize to unit norm: $v = w_\ell / \|w_\ell\|$.
- For scaling factors $\alpha \in \{-2\sigma_y, -\sigma_y, 0, +\sigma_y, +2\sigma_y\}$ where $\sigma_y$ is the standard deviation of $y$ across the test set:
  - Forward pass with $\phi_\ell \mapsto \phi_\ell + \alpha v$.
  - Record per-class logit changes $\Delta z_c(\alpha)$.
- Check monotonicity: $\partial \Delta z_c / \partial \alpha$ should have a sign predictable from theory (e.g., increasing $C_{40}$ should push logits *away* from PSK classes and *toward* QAM classes).

### 7.4 Matched-Pair Counterfactual Analysis
- From WCA-Set, sample matched pairs $(x_i, x_j)$ with same class, same multipath, same CFO, but different SNR (so $y_{\text{DSP}}(x_i) \neq y_{\text{DSP}}(x_j)$).
- Compute cosine distance between $\phi_\ell(x_i)$ and $\phi_\ell(x_j)$ projected onto the probe direction $w_\ell^{(y)}$.
- Check Spearman correlation between projected-latent distance and $|y_{\text{DSP}}(x_i) - y_{\text{DSP}}(x_j)|$; expect $\rho > 0.5$ for represented features.

---

## 8. Standard vs Adversarially-Trained Comparison

### 8.1 Hypotheses
- **H1** (representation): AT-ResNet shows higher linear $R^2$ for Tier-B and Tier-C features (cyclostationary and HOC) than standard ResNet-33, at matched layer depths.
- **H2** (alignment): AT-ResNet shows smaller Linear Gap $\Delta_\ell$ for the same features.
- **H3** (causal use): LEACE erasure of $C_{40}$ and $|R^\alpha_x|$ in AT-ResNet produces a larger and more feature-specific accuracy drop on dependent modulation pairs than in standard ResNet-33.

### 8.2 Statistical Testing
- For each (feature, layer), compare $R^2$ across 3 seeds × 3 split seeds = 9 measurements per model.
- Paired Wilcoxon signed-rank test between standard and AT models.
- Bonferroni correction across the (feature × layer) family.

### 8.3 Mechanistic Interpretation
If H1–H3 are jointly supported, this constitutes a quantitative, latent-space confirmation of the qualitative observation by Maroto et al. that adversarial training drives AMC representations toward maximum-likelihood-like behavior — extending it from constellation-space visualization to physics-aligned internal coding.

---

## 9. Anticipated Figures and Tables

| # | Type | Content |
|---|---|---|
| F1 | Heat-map | $R^2$ (DSP feature × layer) for each model |
| F2 | Curves | Selectivity $S_\ell(y) = R^2_{\text{true}} - R^2_{\text{control}}$ vs. layer depth |
| F3 | Curves | Linear Gap $\Delta_\ell(y)$ vs. layer depth, with control-task gap subtracted |
| F4 | Bar chart | Mean within-class $R^2$ (WCA-Set) vs. between-class $R^2$ (RML) per feature |
| F5 | Curves | Probe $R^2$ vs. SNR, overlaid with modulation classification accuracy curve |
| F6 | Bar chart | LEACE-induced accuracy drop on dependent vs independent modulation pairs |
| F7 | Curves | Steering vector $\alpha$ vs. $\Delta z_c$ for each modulation class |
| F8 | Scatter | Matched-pair latent-distance vs. DSP-feature-distance |
| F9 | Heat-map | AT vs Standard model: $\Delta R^2$ (per feature, per layer) |
| T1 | Table | Validity checklist (V1, V2, V3) pass/fail for each (model, feature) |
| T2 | Table | Paired statistical tests for H1, H2, H3 |

---

## 10. Validation Checklist

Before drawing any conclusions, confirm:

- [ ] All three models reach the accuracy thresholds in §4.4.
- [ ] Ground-truth DSP feature distributions are visualized (histogram per class); features with degenerate distribution within a class are noted but not used for V2.
- [ ] Estimator-noise ceiling is reported alongside each probe $R^2$.
- [ ] Probe $\lambda$ selection is verified via CV (not a single held value).
- [ ] Random-seed variance across 3 model seeds and 3 split seeds is $< 0.05$ in $R^2$ — if not, increase data or revisit splits.
- [ ] Splits are session/seed-based, not frame-level random.
- [ ] V1 (selectivity), V2 (within-class), V3 (causal) are each evaluated and reported; claims of "explicit representation" require all three.
- [ ] LEACE and INLP agree within $\pm 2$ pp on the headline ablation result.
- [ ] Steering-vector effects are monotonic in $\alpha$; non-monotonic curves are reported as evidence of non-linear encoding.
- [ ] All adversarial-vs-standard claims (H1–H3) survive Bonferroni correction.

---

## 11. Implementation Notes

- **Framework:** PyTorch 2.x; activations cached to disk as `safetensors`.
- **Probes:** `scikit-learn` (`Ridge`, `LogisticRegression`) for linear; pure PyTorch for MLP.
- **LEACE / INLP:** `concept-erasure` package; `nullspace-projection` reference repo for INLP.
- **Cyclostationary estimators:** custom implementation of TSM and FAM, validated against the canonical figures in Spooner's blog ("My Papers [42]") and Antoni (*Mech. Syst. Signal Process.* 2009) for spectral kurtosis.
- **Compute estimate:**
  - Training 3 models: ~24 GPU-hours on one A6000.
  - Activation caching: ~12 GPU-hours.
  - Probing (linear + MLP) over all (model, layer, feature) combinations × 3 seeds: ~40 CPU-hours, embarrassingly parallel.
  - LEACE ablations: ~12 GPU-hours.
  - Steering / counterfactual analysis: ~4 GPU-hours.
  - **Total: ~52 GPU-hours + 40 CPU-hours**, comfortably feasible in a 1-month window on a single workstation.
- **Reproducibility:** all seeds, hyperparameters, and the WCA-Set generation script will be archived; activations and probes are deterministic given the seeds.

---

## 12. References

1. Alain, G. & Bengio, Y. (2016). *Understanding intermediate layers using linear classifier probes.* arXiv:1610.01644.
2. Antoni, J. (2009). *Cyclostationarity by examples.* Mech. Syst. Signal Process., 23(4), 987–1036.
3. Belinkov, Y. (2022). *Probing Classifiers: Promises, Shortcomings, and Advances.* Computational Linguistics, 48(1).
4. Belrose, N., Furman, D., Smith, L., Halawi, D., Ostrovsky, I., McKinney, L., Biderman, S., Steinhardt, J. (2023). *LEACE: Perfect Linear Concept Erasure in Closed Form.* NeurIPS.
5. Bricken, T. et al. (2023). *Towards Monosemanticity: Decomposing Language Models with Dictionary Learning.* Anthropic.
6. Conmy, A., Mavor-Parker, A. N., Lynch, A., Heimersheim, S., Garriga-Alonso, A. (2023). *Towards Automated Circuit Discovery for Mechanistic Interpretability.* NeurIPS.
7. Hewitt, J. & Liang, P. (2019). *Designing and Interpreting Probes with Control Tasks.* EMNLP.
8. Maroto, J., Bovet, G., Frossard, P. (2022). *SafeAMC: Adversarial Training for Robust Modulation Classification Models.* EUSIPCO (arXiv:2105.13746, 2021).
9. O'Shea, T. J. & West, N. (2016). *Radio Machine Learning Dataset Generation with GNU Radio.* Proc. GNU Radio Conf.
10. Ravfogel, S., Elazar, Y., Gonen, H., Twiton, M., Goldberg, Y. (2020). *Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection.* ACL.
11. Snoap, J. A., Popescu, D. C., Latshaw, J. A., Spooner, C. M. (2023). *Deep-Learning-Based Classification of Digitally Modulated Signals Using Capsule Networks and Cyclic Cumulants.* Sensors.
12. Snoap, J. A. & Spooner, C. M. (2024). *Novel Neural-Network Preprocessing Layers for Modulation Classification That Generalizes.* IEEE Trans. on Broadcasting.
13. Syed, A., Rager, C., Conmy, A. (2023). *Attribution Patching Outperforms Automated Circuit Discovery.* arXiv:2310.10348.
