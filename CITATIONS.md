# Citations and Source Methods

`clinikit` is an **integration toolkit**. The 14 hybrid classifiers
and supporting utilities it ships are adaptations of techniques
published in the academic and open-source machine-learning
literature. This file lists the source methods, the published work
they derive from, and the open-source libraries `clinikit` builds on
top of.

> If a citation is missing or incorrect, please open an issue or
> pull request — accurate attribution is a hard requirement.

---

## 1. Classifiers (`clinikit.models`)

### `RuleAugmentedClassifier`

Combines a learned probabilistic model with explicit symbolic rules
applied at inference time.

- Hu, Z., Ma, X., Liu, Z., Hovy, E., & Xing, E. (2016).
  *Harnessing deep neural networks with logic rules.*
  Proceedings of ACL 2016.
- Friedman, J. H., & Popescu, B. E. (2008).
  *Predictive learning via rule ensembles.*
  The Annals of Applied Statistics, 2(3), 916-954.

### `BoundaryRefineClassifier`

Refines the decision boundary of a base learner using a secondary
margin-aware learner on examples near the boundary.

- Cortes, C., & Vapnik, V. (1995). *Support-vector networks.*
  Machine Learning, 20(3), 273-297.
- Bartlett, P. L., & Wegkamp, M. H. (2008).
  *Classification with a reject option using a hinge loss.*
  Journal of Machine Learning Research, 9(Aug), 1823-1840.

### `SubgroupThresholdClassifier`

Learns subgroup-specific decision thresholds to balance error rates
across subpopulations.

- Hardt, M., Price, E., & Srebro, N. (2016).
  *Equality of opportunity in supervised learning.*
  NeurIPS 2016.
- Pleiss, G., Raghavan, M., Wu, F., Kleinberg, J., & Weinberger, K. Q.
  (2017). *On fairness and calibration.* NeurIPS 2017.

### `ErrorAwareCalibrator`

Post-hoc calibrator that conditions on error-prone regions of the
input space, extending Platt scaling and isotonic regression.

- Platt, J. (1999). *Probabilistic outputs for support vector machines
  and comparisons to regularized likelihood methods.*
- Niculescu-Mizil, A., & Caruana, R. (2005). *Predicting good
  probabilities with supervised learning.* ICML 2005.
- Kull, M., Filho, T. S., & Flach, P. (2017). *Beta calibration.*
  AISTATS 2017.

### `MonotonicBooster`

Gradient-boosted trees with user-specified monotonic constraints
per feature.

- Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting
  system.* KDD 2016.
- Bartley, C., Liu, W., & Reynolds, M. (2019). *Effective monotone
  knowledge integration in kernel support vector machines.*

### `HardSampleWeightedEnsemble`

Ensemble whose constituent models receive higher sample weights on
historically misclassified instances.

- Freund, Y., & Schapire, R. E. (1997). *A decision-theoretic
  generalization of on-line learning and an application to boosting.*
  Journal of Computer and System Sciences, 55(1), 119-139.
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017).
  *Focal loss for dense object detection.* ICCV 2017.

### `ClassConditionalImputer`

Imputes missing values using class-conditional distributions
estimated from the training set.

- Little, R. J. A., & Rubin, D. B. (2019). *Statistical analysis with
  missing data* (3rd ed.). Wiley.
- van Buuren, S. (2018). *Flexible imputation of missing data*
  (2nd ed.). CRC Press.

### `CrossDistributionDistiller`

Knowledge distillation across two related but distinct training
distributions to improve robustness.

- Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the
  knowledge in a neural network.* NeurIPS Deep Learning Workshop.
- Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F.,
  & Vaughan, J. W. (2010). *A theory of learning from different
  domains.* Machine Learning, 79, 151-175.

### `SelectiveClassifier`

A classifier that may abstain on low-confidence inputs, trading
coverage for selective accuracy.

- El-Yaniv, R., & Wiener, Y. (2010). *On the foundations of noise-free
  selective classification.* Journal of Machine Learning Research, 11.
- Geifman, Y., & El-Yaniv, R. (2017). *Selective classification for
  deep neural networks.* NeurIPS 2017.

### `InstanceAdaptiveThreshold`

Predicts a per-instance decision threshold conditional on the input
features and the base classifier's probability output.

- Lipton, Z. C., Elkan, C., & Naryanaswamy, B. (2014). *Optimal
  thresholding of classifiers to maximize F1 measure.*
  ECML-PKDD 2014.
- Koyejo, O. O., Natarajan, N., Ravikumar, P. K., & Dhillon, I. S.
  (2014). *Consistent binary classification with generalized
  performance metrics.* NeurIPS 2014.

### `DialecticalEnsemble`

Ensemble of two adversarially-trained sub-models whose disagreement
is resolved by a tie-breaker arbiter.

- Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley,
  D., Ozair, S., Courville, A., & Bengio, Y. (2014).
  *Generative adversarial nets.* NeurIPS 2014.
- Mukherjee, S., & Schapire, R. E. (2013). *A theory of multiclass
  boosting.* Journal of Machine Learning Research, 14, 437-497.

### `LatentSubtypeRouter`

Mixture-of-experts router that first clusters inputs into latent
subtypes, then dispatches to a subtype-specialized expert.

- Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991).
  *Adaptive mixtures of local experts.* Neural Computation, 3(1).
- Shazeer, N., Mirhoseini, A., Maziarz, K., et al. (2017).
  *Outrageously large neural networks: The sparsely-gated
  mixture-of-experts layer.* ICLR 2017.

### `IterativeLabelRefiner`

Self-training loop that revises noisy labels using model confidence
across iterations.

- Northcutt, C. G., Jiang, L., & Chuang, I. L. (2021).
  *Confident learning: Estimating uncertainty in dataset labels.*
  Journal of Artificial Intelligence Research, 70, 1373-1411.
- Reed, S., Lee, H., Anguelov, D., et al. (2015). *Training deep
  neural networks on noisy labels with bootstrapping.* ICLR Workshop.

### `DualViewCoTrainer`

Co-training of two classifiers, each looking at a different feature
view of the same instances.

- Blum, A., & Mitchell, T. (1998). *Combining labeled and unlabeled
  data with co-training.* COLT 1998.
- Nigam, K., & Ghani, R. (2000). *Analyzing the effectiveness and
  applicability of co-training.* CIKM 2000.

---

## 2. Supporting modules

### `metrics`, `curves`

- Powers, D. M. W. (2011). *Evaluation: from precision, recall and
  F-measure to ROC, informedness, markedness and correlation.*
- Vickers, A. J., & Elkin, E. B. (2006). *Decision curve analysis:
  a novel method for evaluating prediction models.* Medical Decision
  Making, 26(6), 565-574.

### `calibration`

- Platt scaling: Platt (1999).
- Isotonic regression: Zadrozny, B., & Elkan, C. (2002).
  *Transforming classifier scores into accurate multiclass
  probability estimates.* KDD 2002.
- Temperature scaling: Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q.
  (2017). *On calibration of modern neural networks.* ICML 2017.

### `statistics`

- DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988).
  *Comparing the areas under two or more correlated receiver
  operating characteristic curves: a nonparametric approach.*
  Biometrics, 44(3), 837-845.
- Efron, B., & Tibshirani, R. J. (1993). *An introduction to the
  bootstrap.* Chapman & Hall.
- McNemar, Q. (1947). *Note on the sampling error of the difference
  between correlated proportions or percentages.* Psychometrika, 12.

### `diagnostics`

- Cleanlab: Northcutt, C. G., et al. (2021).
- Influence functions: Koh, P. W., & Liang, P. (2017).
  *Understanding black-box predictions via influence functions.*
  ICML 2017.

### `cost_sensitive`

- Elkan, C. (2001). *The foundations of cost-sensitive learning.*
  IJCAI 2001.

### `monitor`

- Kolmogorov-Smirnov drift: standard non-parametric two-sample test.
- Population Stability Index: industry standard, see Karakoulas (2004),
  *Empirical validation of retail credit-scoring models.*
- Wasserstein distance: Kantorovich-Rubinstein duality, standard.

### `modelcard`

- Mitchell, M., et al. (2019). *Model cards for model reporting.*
  FAT* 2019.

### `external_val`

- Steyerberg, E. W. (2019). *Clinical prediction models: a practical
  approach to development, validation, and updating* (2nd ed.).
  Springer.

### `explainability`

- Lundberg, S. M., & Lee, S.-I. (2017). *A unified approach to
  interpreting model predictions.* NeurIPS 2017 (SHAP).
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016).
  *"Why should I trust you?": Explaining the predictions of any
  classifier.* KDD 2016 (LIME).

### `automl`

- TabPFN: Hollmann, N., Müller, S., Eggensperger, K., & Hutter, F.
  (2023). *TabPFN: A transformer that solves small tabular
  classification problems in a second.* ICLR 2023.
- FLAML: Wang, C., Wu, Q., Weimer, M., & Zhu, E. (2021).
  *FLAML: A fast and lightweight AutoML library.* MLSys 2021.
- AutoGluon: Erickson, N., et al. (2020). *AutoGluon-Tabular: Robust
  and accurate AutoML for structured data.*

### `synthetic`

- Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K.
  (2019). *Modeling tabular data using conditional GAN.* NeurIPS 2019
  (CTGAN / TVAE).

### `active_learning`

- Settles, B. (2009). *Active learning literature survey.*
  University of Wisconsin-Madison.

---

## 3. Open-source libraries we depend on

| Library          | Role                                          | License        |
| ---------------- | --------------------------------------------- | -------------- |
| numpy            | Array primitives                              | BSD-3-Clause   |
| pandas           | Tabular data                                  | BSD-3-Clause   |
| scikit-learn     | Base estimators, validation, transformers     | BSD-3-Clause   |
| scipy            | Statistical primitives                        | BSD-3-Clause   |
| xgboost          | Gradient boosting                             | Apache-2.0     |
| lightgbm         | Gradient boosting                             | MIT            |
| catboost         | Gradient boosting                             | Apache-2.0     |
| imbalanced-learn | Resampling                                    | MIT            |
| matplotlib       | Plotting                                      | matplotlib     |
| typer            | CLI                                           | MIT            |
| jinja2           | HTML report templates                         | BSD-3-Clause   |
| joblib           | Model serialization                           | BSD-3-Clause   |

Optional dependencies follow the same permissive licensing
(`shap`, `lime`, `cleanlab`, `mapie`, `tabpfn`, `flaml`, `autogluon`,
`sdv`, `ngboost`, `interpret`, `feature-engine`, `modAL`).

---

## 4. Datasets

`clinikit` ships small UCI benchmark datasets for tests and tutorials:

- PIMA Indians Diabetes — UCI Machine Learning Repository.
- Wisconsin Breast Cancer (Diagnostic) — UCI Machine Learning Repository.
- UCI Heart Disease — UCI Machine Learning Repository, Cleveland subset.
- Frankfurt Diabetes — public benchmark, bundled if license permits.

Each dataset's original source URL and any license note is recorded
inside the `datasets` module's documentation.
