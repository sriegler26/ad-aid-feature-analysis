# ad-aid-feature-analysis

Analysis code for the study of documented clinical features in patients with Alzheimer's disease (AD) with and without autoimmune disease (AID), using clinical notes from the UCSF de-identified electronic health record (EHR).

**No patient-level data are in this repository.** Running the code requires access to the UCSF de-identified EHR (see the Data availability statement of the paper). Every input path in the scripts is blank and must be set to the local copy of the input tables.

## Contents

```
analysis/    computes every statistic and writes the source-data workbooks (.xlsx)
products/    draws the tables (.docx) and figures (.png, .pdf) from those workbooks
prompt/       the LLM extraction prompt (Supplementary Data 1)
```

The two folders are separate on purpose:
- The analysis scripts never format a number.
- The table and figure scripts never fit a model or compute a statistic. Every value they show is read from a workbook.

Shared code:
- `analysis/analysis_utils.py`: the loaders and their checks, the diagnosis-day split, the age ceiling, the floor checks, standardized mean differences, Benjamini–Hochberg FDR, Wilson intervals, indirect standardization, and the workbook writers.
- `products/render_utils.py`: the single number formatter and the Word table writer.
- `write_guard.py`: the only code that writes a file. An identical copy sits in each folder.

## Requirements

Python 3.12 with:
- pandas 3.1.1
- numpy 2.5.4
- scipy 1.18.1
- statsmodels 0.15.0
- scikit-learn 1.9.1
- matplotlib 3.11.2
- openpyxl, pyarrow, and python-docx

## Write safety

The scripts never alter, overwrite, move, or delete an existing file.
- Every output is a new file under a single output root, `OUTPUT_ROOT_DIR` in `write_guard.py`. The root is relative to the data folder the scripts are run from.
- The root is created once, by `write_guard.create_root()`, together with a marker file. An existing folder is never reused.
- A script stops with `WriteRefused` if an output already exists, if a path lies outside the root, or if the marker is missing.

To run again, rename or move the earlier output root yourself first.

## Setup

1. Keep this code folder outside your data folder.
2. Set `OUTPUT_ROOT_DIR` in both copies of `write_guard.py` (`analysis/` and `products/`) to the same value.
3. Fill in the blank `*_PATH` and `*_DIR` settings at the top of each script. Each one is commented with the file format and the content the script expects.
4. Create the output root once, from the data folder:

   ```
   cd DATA_FOLDER
   python -c "import sys; sys.path.insert(0, 'CODE/analysis'); import write_guard; write_guard.create_root()"
   ```

## Run order

Run every script from the data folder as `python CODE/<folder>/<script>.py`.

**Analysis (`analysis/`)**

1. `propensity_score.py` runs first. Every weighted analysis reads the weights it writes.
2. `dlb_core_analysis.py` runs before `aid_condition_analysis.py`. The latter reads the discriminant coefficients written by the former.
3. The rest, in any order:
   - `cohort_characteristics.py`
   - `symptom_prevalence.py`
   - `symptom_timing_sequence.py`
   - `inflammation_tier_sensitivity.py`
   - `cognitive_score_comparison.py`
   - `documentation_language_checks.py`
   - `medication_analysis.py`
   - `mci_diagnosis_dates.py`
   - `dlb_feature_timing.py`

**Tables and figures (`products/`)**

Run every `*_tables.py` and `*_figure.py` script, in any order, after the analysis script whose workbooks it reads. With a blank caption path, the scripts write the bare tables and figures.

## Script map

| Analysis script | What it computes | Table and figure scripts |
|---|---|---|
| `propensity_score.py` | Documentation-balance propensity-score weights, covariate balance, and weighting checks | `propensity_score_balance_figure.py`, `covariate_balance_tables.py` |
| `cohort_characteristics.py` | Cohort characteristics (Table 1, by sex, patients without notes) and coded Lewy body diagnoses | `cohort_characteristics_tables.py` |
| `symptom_prevalence.py` | Symptom prevalence, AID+ versus AID−, weighted and unweighted, with sex interactions | `symptom_prevalence_tables.py` |
| `symptom_timing_sequence.py` | Age at first documentation and the expected order of first documentation | `symptom_timing_sequence_tables.py` |
| `inflammation_tier_sensitivity.py` | Sensitivity analyses by inflammation tier | `inflammation_tier_sensitivity_tables.py` |
| `cognitive_score_comparison.py` | Annual change in MMSE and MoCA scores | `cognitive_score_comparison_tables.py` |
| `medication_analysis.py` | Medication exposure analyses | `medication_tables.py` |
| `mci_diagnosis_dates.py` | MCI and AD diagnosis dates | `mci_diagnosis_dates_tables.py` |
| `dlb_core_analysis.py` | DLB features (McKeith 2017), sex heterogeneity, core-feature co-occurrence, the coded DLB versus AD discriminant, and the feature-burden score | `dlb_core_tables.py`, `dlb_feature_specificity_figure.py`, `dlb_core_feature_cooccurrence_figure.py` |
| `dlb_feature_timing.py` | Timing of the DLB features relative to the AD diagnosis | `dlb_feature_timing_tables.py` |
| `aid_condition_analysis.py` | Core DLB features by coded AID condition (indirect standardization), possible mimics, published groupings, and Sjögren's syndrome timing | `aid_condition_tables.py`, `aid_condition_figure.py`, `sjogrens_feature_timing_figure.py` |
| `documentation_language_checks.py` | Checks of change language and explicit-time records (checks workbooks only) | – |
| – | Extraction validation | `extraction_validation_tables.py` |

## Clinical feature extraction

Symptoms were extracted from clinical notes by a large language model (GPT-4.1 mini, deployment `gpt-4.1-mini-2025-04-14`, temperature 0) in the UCSF Versa Azure environment.
- The prompt asks for 56 categories (55 plus "other") and returns JSON records with name, category, source, evidence, and confounders.
- The full prompt is Supplementary Data 1, at `/prompt`.

## License

MIT. See `LICENSE`.
