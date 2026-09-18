# Methodological notes

These notes describe a retrospective code review. They do not claim that the following improvements were part of the original experiment.

## More independent within-subject evaluation

Create disjoint raw time blocks before windowing and leave a gap between partitions. Explicitly define which preprocessing operations can use test-recording information. Prefer independent sessions where available. Compare the new results with the saved baseline rather than assuming that all of the current high accuracy is caused by overlap.

## Cross-subject evaluation

Report the distribution of per-subject scores and subject-level uncertainty, not only a mean. Verify model fit logs: the current exception handlers can replace a failed model with all-negative predictions. Use an inner subject-grouped validation loop if tuning hyperparameters, leaving the outer test subject untouched. Subject adaptation and calibration are possible future experiments, not completed results.

## Statistical analysis

The exploratory tests pool channels and windows and treat them as independent observations. Consider per-subject condition summaries with paired tests, or an appropriate repeated-measures model. Address multiplicity and effect sizes. Significance annotations in the existing figures and CSV are retained as historical exploratory output, not endorsed as confirmatory population-level evidence.

## Feature interpretation

For band power P and a fixed window of N samples, the implementation computes DE = 0.5 log2(2 pi e P) and LEE = log10(N P). Excluding numerical lower-bound clipping, these carry equivalent information through an affine transformation. They should not be presented as independent complexity measures. The DE formula assumes the Gaussian entropy relationship; it is not a general entropy estimator.

## Construct validity and source preprocessing

The binary labels distinguish rest and arithmetic recordings. They do not by themselves isolate task difficulty or establish a cognitive-load scale. The public data were already cleaned by the providers, and the acquisition-filter wording in the source requires clarification before gamma-band interpretation. Confirm original channel order and preprocessing provenance when reproducing the pipeline.

## Publication scope

This repository contains six analysis scripts, CSV result exports and selected existing figures. The original local project also contains large data/intermediate files, personal coursework reports, application-writing notes and unrelated fuzzy-logic exercises. Those materials are outside this release. Path edits make the scripts portable; no full MATLAB execution was performed during publication.
