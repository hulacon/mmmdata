"""Condition-level GLM for MMMData: BIDS Stats Models in, contrast maps out.

The skeleton of the glm-strategy architecture (mmmdata-agents
``docs/workbench/glm-strategy/log.md``, DECIDED 2026-08-25):

1. **Declaration** — a BIDS Stats Model JSON in ``<repo>/models/`` names the
   task, the conditions, and the contrasts (:mod:`.models`). It is the
   Contract C join surface for models; it says nothing about estimators.
2. **Design** — events + fMRIPrep confounds become a nilearn design matrix
   (:mod:`.design`), with the one shared :class:`~.config.GlmConfig` supplying
   the space, HRF, confound set and noise model that per-script constants
   used to agree on only by hand.
3. **Estimation** — an :class:`~.estimators.Estimator` maps (design, data,
   covariance model) to (effect, variance, dof) per contrast, so the nilearn
   wrapper, a REMLfit wrapper and braintwill's GLS core can be benchmarked
   through one interface. Runs pool by precision-weighted fixed effects.
4. **Outputs** — statistical maps named with Contract A keys
   (:mod:`.outputs`) so the catalog can find them.
5. **Reliability** — split-half Dice on top-N masks (:mod:`.reliability`),
   the charter's bake-off metric.

Nothing here has run on real data yet; the first real fit is a cluster step
(cluster-reentry R15). ``scripts/glm_contrast_maps.py`` is the runner.
"""
