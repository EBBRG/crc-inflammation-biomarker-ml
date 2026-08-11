# CRC Inflammation Biomarker Discovery Pipeline — Advanced 2026

### A Publication-Ready Multi-Omics Machine-Learning Pipeline for Colorectal Cancer Inflammation Biomarker Discovery

The CRC Inflammation Biomarker Discovery Pipeline integrates **classical machine learning**, a **novel Gene-Attention Transformer (GAT)**, and **SHAP-based explainability** into a unified 5-dimensional consensus-scoring framework that discovers, validates, and clinically annotates inflammation biomarkers for colorectal cancer (CRC). It performs fully automated external validation on independent GEO and TCGA cohorts and includes bootstrap stability analysis and survival analysis — all in a single reproducible run.

Developed and maintained by the **Evo Biology and Bioinformatics Research Group (EBBRG)**, University of Agriculture Faisalabad.

---

## Table of Contents

- [Novel Contributions](#novel-contributions)
- [Features](#features)
- [Pipeline Architecture](#pipeline-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Input Files](#input-files)
- [Output Structure](#output-structure)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Novel Contributions

| # | Novel Method | What it does | Why it's novel |
|---|---|---|---|
| 1 | **Gene-Attention Transformer (GAT)** | Treats each gene as a token, learns gene–gene co-expression relationships via multi-head self-attention | First application of biological-positional-encoding Transformer to CRC inflammation gene selection |
| 2 | **SHAP-weighted 5D Consensus Scoring** | Replaces vote-counting with a quantitative composite score across 5 biological dimensions | Combines method agreement + SHAP + attention + effect size + prior knowledge into one score |
| 3 | **Automated External Validation** | Downloads GEO cohorts (GSE39582, GSE17536, GSE14333) and validates the panel automatically | Zero manual steps — fully reproducible pipeline |

---

## Features

- **4 complementary feature selectors** — ElasticNet-CV, SVM-RFE, Random Forest (permutation importance), and the novel GAT
- **Explainable consensus** — 5D composite scoring: method agreement, SHAP rank, attention weight, Cohen's *d* effect size, inflammation relevance
- **Rigorous validation** — nested cross-validation across 5 classifiers, bootstrap stability (n=1000, 95% CI), automated GEO + TCGA external validation
- **Survival analysis** — Kaplan–Meier curves and Cox proportional-hazards on the discovered panel
- **Fully automated** — one command runs the entire pipeline end-to-end
- **Reproducible** — fixed configuration, structured outputs, machine-readable summary

---

## Pipeline Architecture

```
expression_matrix.tsv
      │
      ▼
DataLoader → DataPreparation (filter, scale, split)
      │
      ├─── [Classical] LassoSelector (ElasticNet-CV)
      ├─── [Classical] SVMRFESelector (RFECV)
      ├─── [Classical] RandomForestSelector (permutation importance)
      └─── [NOVEL] TransformerGeneSelector (Gene-Attention Transformer)
                          │
                          ▼
               [NOVEL] SHAPExplainer (RF + GBM + LR SHAP values)
                          │
                          ▼
          [NOVEL] AdvancedConsensusIdentifier
          5-dimensional biomarker scoring:
            (1) method agreement
            (2) SHAP rank
            (3) attention weight
            (4) Cohen's d effect size
            (5) inflammation relevance
                          │
                          ▼
               AdvancedModelValidator (5 classifiers, nested CV)
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
   BootstrapStabilityAnalyzer   ExternalValidator
   (1000 iterations, 95% CI)    (GEO + TCGA auto-download)
              │
              ▼
      SurvivalAnalyzer (KM + Cox PH)
              │
              ▼
      FinalReporter + Figures
```

---

## Installation

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0 (CPU or GPU)
- See `requirements.txt` for the full stack

```bash
git clone https://github.com/EBBRG/crc-inflammation-biomarker-ml.git
cd crc-inflammation-biomarker-ml
pip install -r requirements.txt
```

---

## Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python crc_advanced_part3.py
```

---

## Input Files

| File | Format | Description |
|------|--------|-------------|
| `expression_matrix.tsv` | TSV, genes × samples | Log2 normalized expression |
| `sample_groups.csv` | CSV | Columns: sample_id, group (Tumor/Normal), survival_days, event, stage |
| `inflammation_genes.txt` | TXT | One gene symbol per line |
| `immune_correlation.csv` | CSV (optional) | gene, immune_cell, correlation |
| `GO_BP_results.csv` | CSV (optional) | GO enrichment results |
| `KEGG_results.csv` | CSV (optional) | KEGG enrichment results |

---

## Output Structure

```
CRC_Advanced_Results_2026/
  Tables/
    CONSENSUS_BIOMARKER_PANEL.csv     ← Primary result
    ALL_GENE_SCORES.csv               ← 5D scores for all genes
    MODEL_PERFORMANCE.csv             ← AUC, PR-AUC per model
    BOOTSTRAP_GENE_STABILITY.csv      ← Selection frequency
    BOOTSTRAP_AUC_CI.csv              ← 95% confidence interval
    EXTERNAL_VALIDATION.csv           ← GEO/TCGA AUCs
    SURVIVAL_ANALYSIS.csv             ← HR, logrank p, C-index
  Figures/
    GAT_01_training_curves            ← Transformer training
    SHAP_01_importance_per_model      ← SHAP bar charts
    CONSENSUS_01_score_heatmap        ← 5D scoring heatmap
    CONSENSUS_02_method_comparison    ← Overlap bar chart
    VALIDATION_01_ROC_curves          ← Multi-model ROC
    VALIDATION_02_PR_curves           ← Precision-Recall
    BOOTSTRAP_01_stability            ← Stability + CI
    ATTENTION_01_visualization        ← Transformer attention maps
    EXTERNAL_01_validation_forest     ← Forest plot of AUCs
    SURVIVAL_01_kaplan_meier          ← KM curves + HR
  Models/
    gene_attention_transformer.pt     ← Saved Transformer weights
  Reports/
    pipeline_summary.json             ← Machine-readable summary
```

---

## Configuration

Key config options (edit in `crc_advanced_part1.py`):

```python
CFG.EXTERNAL_DATASETS  = ["GSE39582", "GSE17536", "GSE14333"]  # GEO cohorts to validate on
CFG.N_BOOTSTRAP        = 1000       # Bootstrap iterations
CFG.TRANSFORMER_EPOCHS = 80         # Increase for better GAT training
CFG.MIN_CONSENSUS      = 2          # Minimum methods agreeing for consensus
CFG.SHAP_TOP_N         = 50         # Top N genes to compute SHAP for
CFG.USE_TCGA           = True       # Enable/disable TCGA validation
```

---

## Dependencies

| Package | Purpose |
|---|---|
| numpy, pandas, scipy, statsmodels | Core scientific stack |
| scikit-learn | Classical ML, nested CV |
| torch | Gene-Attention Transformer |
| shap | Explainability |
| lifelines | Survival analysis |
| GEOparse, requests | GEO external validation |
| matplotlib, seaborn | Figures |
| tqdm | Progress bars |

---

## Repository Structure

```
crc-inflammation-biomarker-ml/
│
├── crc_advanced_part1.py   # Config, data prep, classical ML + GAT selection
├── crc_advanced_part2.py   # SHAP, consensus scoring, model validation
├── crc_advanced_part3.py   # External validation + survival + master runner
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Citation

If you use this pipeline in your research, please cite the EBBRG group and link to this repository:

> Evo Biology and Bioinformatics Research Group (EBBRG). *CRC Inflammation Biomarker Discovery Pipeline — Advanced 2026.* University of Agriculture Faisalabad. https://github.com/EBBRG/crc-inflammation-biomarker-ml

### Methods section template

> Feature selection was performed using four complementary approaches: ElasticNet (Tibshirani 1996), SVM-RFE (Guyon et al. 2002), Random Forest with permutation importance (Breiman 2001), and a novel Gene-Attention Transformer (GAT) that treats each gene as a sequence token and applies multi-head self-attention with biological positional encoding to identify co-expression-aware gene importance scores.
>
> Consensus biomarkers were identified using a 5-dimensional composite scoring framework incorporating method agreement, SHAP-based model-agnostic importance (Lundberg & Lee 2017), transformer attention weights, Cohen's d effect size, and prior inflammation gene relevance.
>
> Model performance was assessed by nested cross-validation, bootstrap resampling (n=1000, 95% CI), and external validation on three independent GEO cohorts (GSE39582, GSE17536, GSE14333).

---

## License

Released under the **MIT License**. See `LICENSE`.

---

## Contact

**Evo Biology and Bioinformatics Research Group (EBBRG)**
University of Agriculture Faisalabad, Pakistan

For questions, bug reports, or feature requests, please use the GitHub issue tracker.
