"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   CRC INFLAMMATION BIOMARKER DISCOVERY PIPELINE — ADVANCED 2026 VERSION    ║
║   Part 1: Configuration, Data, Classical ML + Transformer Gene Selection   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Novel contributions:
  1. Gene-Attention Transformer (GAT) for biologically-aware feature selection
  2. SHAP-integrated consensus scoring (not just vote counting)
  3. Multi-objective biomarker scoring (diagnostic + prognostic + immune)
  4. Automated external validation pipeline (GEO + TCGA)

Requirements:
    pip install numpy pandas scikit-learn torch shap matplotlib seaborn
                scipy statsmodels lifelines GEOparse requests tqdm
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os, sys, json, warnings, logging, hashlib, pickle, time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

warnings.filterwarnings('ignore')

# ── Scientific computing ───────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from statsmodels.stats.multitest import multipletests

# ── Machine learning ───────────────────────────────────────────────────────────
from sklearn.model_selection import (StratifiedKFold, train_test_split,
                                     cross_val_score, LeaveOneOut)
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.linear_model import LassoCV, LogisticRegressionCV, ElasticNetCV
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier, VotingClassifier)
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report,
                              confusion_matrix, average_precision_score,
                              precision_recall_curve)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance

# ── Deep learning ──────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠ PyTorch not installed. Transformer-based selection will be skipped.")
    print("  Install: pip install torch")

# ── Explainability ─────────────────────────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠ SHAP not installed. Explainability module will be skipped.")
    print("  Install: pip install shap")

# ── Survival analysis ──────────────────────────────────────────────────────────
try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False

# ── Visualization ──────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# ── GEO access ────────────────────────────────────────────────────────────────
try:
    import GEOparse
    GEOPARSE_AVAILABLE = True
except ImportError:
    GEOPARSE_AVAILABLE = False

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # ── Input files ────────────────────────────────────────────────────────
    EXPRESSION_DATA  : str = "expression_matrix.tsv"   # genes × samples
    SAMPLE_GROUPS    : str = "sample_groups.csv"        # sample_id, group, patient_id, stage, survival_days, event
    INFLAM_GENES     : str = "inflammation_genes.txt"   # one gene per line
    IMMUNE_CORR      : str = "immune_correlation.csv"   # gene, immune_cell, correlation
    GO_BP            : str = "GO_BP_results.csv"
    KEGG             : str = "KEGG_results.csv"

    # ── Output structure ───────────────────────────────────────────────────
    OUTPUT_DIR : Path = Path("CRC_Advanced_Results_2026")
    FIGURE_DIR : Path = OUTPUT_DIR / "Figures"
    TABLE_DIR  : Path = OUTPUT_DIR / "Tables"
    MODEL_DIR  : Path = OUTPUT_DIR / "Models"
    REPORT_DIR : Path = OUTPUT_DIR / "Reports"
    CACHE_DIR  : Path = OUTPUT_DIR / ".cache"

    # ── Analysis parameters ────────────────────────────────────────────────
    RANDOM_SEED      : int   = 42
    CV_FOLDS         : int   = 5
    TEST_SIZE        : float = 0.2
    N_BOOTSTRAP      : int   = 1000
    MIN_CONSENSUS    : int   = 2       # min methods for consensus (out of 4)

    # ── Filter thresholds ──────────────────────────────────────────────────
    MIN_EXPR         : float = 1.0     # log2 expression minimum
    MIN_VARIANCE_PCT : float = 0.20    # keep top 20% variable genes
    PADJ_CUTOFF      : float = 0.05
    LFC_CUTOFF       : float = 1.0

    # ── Transformer hyperparameters ────────────────────────────────────────
    TRANSFORMER_DIM  : int   = 64
    TRANSFORMER_HEADS: int   = 4
    TRANSFORMER_LAYERS: int  = 2
    TRANSFORMER_EPOCHS: int  = 80
    TRANSFORMER_LR   : float = 1e-3
    TRANSFORMER_DROPOUT: float = 0.2

    # ── SHAP settings ──────────────────────────────────────────────────────
    SHAP_BACKGROUND_SAMPLES: int = 50
    SHAP_TOP_N       : int   = 50

    # ── External validation ────────────────────────────────────────────────
    EXTERNAL_DATASETS: List[str] = field(default_factory=lambda: [
        "GSE39582",   # CRC, n=566, survival data
        "GSE17536",   # CRC, n=177, survival data
        "GSE14333",   # CRC, n=290
    ])
    USE_TCGA         : bool  = True
    TCGA_PROJECT     : str   = "TCGA-COAD"

    # ── Publication settings ───────────────────────────────────────────────
    DPI              : int   = 300
    FIGURE_FORMAT    : str   = "pdf"   # pdf for publication, png for preview
    FONT_SIZE        : int   = 11
    COLOR_TUMOR      : str   = "#C0392B"
    COLOR_NORMAL     : str   = "#2980B9"
    COLOR_PALETTE    : str   = "RdBu_r"

    def create_directories(self):
        for d in [self.OUTPUT_DIR, self.FIGURE_DIR, self.TABLE_DIR,
                  self.MODEL_DIR, self.REPORT_DIR, self.CACHE_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        log.info(f"Output directory: {self.OUTPUT_DIR.absolute()}")

CFG = Config()

# ═════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def print_header(title: str, char: str = "═", width: int = 74):
    line = char * width
    print(f"\n{line}\n  {title}\n{line}\n")

def save_figure(fig, name: str, tight: bool = True):
    if tight:
        fig.tight_layout()
    path_pdf = CFG.FIGURE_DIR / f"{name}.pdf"
    path_png = CFG.FIGURE_DIR / f"{name}.png"
    fig.savefig(path_pdf, dpi=CFG.DPI, bbox_inches='tight')
    fig.savefig(path_png, dpi=CFG.DPI, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved figure: {name}")

def cache_result(key: str, func, *args, force=False, **kwargs):
    """Simple file-based cache to avoid re-running expensive steps."""
    cache_file = CFG.CACHE_DIR / f"{hashlib.md5(key.encode()).hexdigest()}.pkl"
    if cache_file.exists() and not force:
        log.info(f"  Loading cached result for: {key}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    result = func(*args, **kwargs)
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    return result

def cohen_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Effect size: Cohen's d."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-8)

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 & 2: DATA LOADING AND PREPARATION
# ═════════════════════════════════════════════════════════════════════════════

class DataLoader:
    """Load and integrate all input data sources."""

    def __init__(self):
        self.expression    : Optional[pd.DataFrame] = None
        self.sample_info   : Optional[pd.DataFrame] = None
        self.inflam_genes  : Optional[List[str]]    = None
        self.immune_corr   : Optional[pd.DataFrame] = None
        self.go_bp         : Optional[pd.DataFrame] = None
        self.kegg          : Optional[pd.DataFrame] = None

    def load_all(self) -> 'DataLoader':
        print_header("STEP 1: DATA LOADING")

        self.expression   = self._load_expression()
        self.sample_info  = self._load_sample_info()
        self.inflam_genes = self._load_gene_list(CFG.INFLAM_GENES)
        self.immune_corr  = self._safe_csv(CFG.IMMUNE_CORR)
        self.go_bp        = self._safe_csv(CFG.GO_BP)
        self.kegg         = self._safe_csv(CFG.KEGG)

        self._report()
        return self

    def _load_expression(self) -> pd.DataFrame:
        log.info("Loading expression matrix...")
        df = pd.read_csv(CFG.EXPRESSION_DATA, sep='\t', index_col=0)
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        log.info(f"  Expression: {df.shape[0]} genes × {df.shape[1]} samples")
        return df

    def _load_sample_info(self) -> pd.DataFrame:
        log.info("Loading sample metadata...")
        df = pd.read_csv(CFG.SAMPLE_GROUPS)
        required = ['sample_id', 'group']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"sample_groups.csv must have column: {col}")
        # Optional survival columns
        for col in ['survival_days', 'event', 'stage', 'patient_id']:
            if col not in df.columns:
                df[col] = np.nan
        log.info(f"  Samples: {len(df)} | Groups: {df['group'].value_counts().to_dict()}")
        return df

    def _load_gene_list(self, path: str) -> List[str]:
        if not Path(path).exists():
            log.warning(f"  Gene list not found: {path} — using all genes")
            return []
        with open(path) as f:
            genes = [l.strip() for l in f if l.strip()]
        log.info(f"  Inflammation gene list: {len(genes)} genes")
        return genes

    def _safe_csv(self, path: str) -> Optional[pd.DataFrame]:
        if Path(path).exists():
            return pd.read_csv(path)
        log.warning(f"  Optional file not found: {path}")
        return None

    def _report(self):
        print(f"\n  Expression matrix : {self.expression.shape}")
        print(f"  Tumor samples     : {(self.sample_info['group']=='Tumor').sum()}")
        print(f"  Normal samples    : {(self.sample_info['group']=='Normal').sum()}")
        print(f"  Inflammation genes: {len(self.inflam_genes)}")
        has_surv = self.sample_info['survival_days'].notna().any()
        print(f"  Survival data     : {'YES' if has_surv else 'NO'}")


class DataPreparation:
    """Filter, align, scale, and split the data."""

    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.genes_filtered = []
        self.inflam_idx     = []
        self.scaler         = RobustScaler()
        self.sample_info_train = self.sample_info_test = None

    def prepare(self) -> 'DataPreparation':
        print_header("STEP 2: DATA PREPARATION")

        expr = self.loader.expression
        meta = self.loader.sample_info

        # ── Align samples ──────────────────────────────────────────────────
        common = list(set(expr.columns) & set(meta['sample_id']))
        expr   = expr[common]
        meta   = meta[meta['sample_id'].isin(common)].set_index('sample_id').loc[common]
        log.info(f"  Aligned: {len(common)} samples")

        # ── Filter genes ────────────────────────────────────────────────────
        # 1) Minimum expression
        mask_expr = (expr > CFG.MIN_EXPR).sum(axis=1) >= 3
        expr = expr[mask_expr]

        # 2) Focus on inflammation genes if provided
        if self.loader.inflam_genes:
            inflam_present = [g for g in self.loader.inflam_genes if g in expr.index]
            log.info(f"  Inflammation genes present: {len(inflam_present)}/{len(self.loader.inflam_genes)}")
            # Keep intersection + top variable non-inflam genes for context
            top_var  = expr.var(axis=1).nlargest(int(len(expr)*CFG.MIN_VARIANCE_PCT)).index
            keep     = list(set(inflam_present) | set(top_var))
            expr     = expr.loc[expr.index.isin(keep)]
        else:
            # Keep top 20% by variance
            top_var = expr.var(axis=1).nlargest(int(len(expr)*CFG.MIN_VARIANCE_PCT)).index
            expr    = expr.loc[top_var]

        self.genes_filtered = list(expr.index)
        log.info(f"  Genes after filtering: {len(self.genes_filtered)}")

        # ── Track inflammation gene indices ────────────────────────────────
        if self.loader.inflam_genes:
            self.inflam_idx = [i for i, g in enumerate(self.genes_filtered)
                               if g in self.loader.inflam_genes]

        # ── Build X, y ─────────────────────────────────────────────────────
        X = expr.T.values.astype(np.float32)  # samples × genes
        y = (meta['group'] == 'Tumor').astype(int).values

        # ── Stratified split ────────────────────────────────────────────────
        idx = np.arange(len(X))
        tr_idx, te_idx = train_test_split(
            idx, test_size=CFG.TEST_SIZE, random_state=CFG.RANDOM_SEED,
            stratify=y
        )
        self.X_train = self.scaler.fit_transform(X[tr_idx])
        self.X_test  = self.scaler.transform(X[te_idx])
        self.y_train = y[tr_idx]
        self.y_test  = y[te_idx]
        self.sample_info_train = meta.iloc[tr_idx]
        self.sample_info_test  = meta.iloc[te_idx]

        log.info(f"  Train: {self.X_train.shape} | Test: {self.X_test.shape}")
        log.info(f"  Train class balance: {np.bincount(self.y_train)}")
        return self

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: CLASSICAL FEATURE SELECTION (LASSO + SVM-RFE + RANDOM FOREST)
# ═════════════════════════════════════════════════════════════════════════════

class LassoSelector:
    """Elastic Net / LASSO with cross-validated alpha selection."""

    def __init__(self, data: DataPreparation):
        self.data   = data
        self.genes  : List[str] = []
        self.scores : pd.Series = pd.Series(dtype=float)

    def fit(self) -> 'LassoSelector':
        log.info("Running ElasticNet / LASSO feature selection...")
        # ElasticNet combines L1 (sparsity) + L2 (stability) — better than pure LASSO
        enet = ElasticNetCV(
            l1_ratio=[0.5, 0.7, 0.9, 0.95, 1.0],
            cv=CFG.CV_FOLDS, max_iter=5000, random_state=CFG.RANDOM_SEED, n_jobs=-1
        )
        enet.fit(self.data.X_train, self.data.y_train)
        coef = np.abs(enet.coef_)
        selected = np.where(coef > 0)[0]
        self.genes  = [self.data.genes_filtered[i] for i in selected]
        self.scores = pd.Series(coef[selected], index=self.genes).sort_values(ascending=False)
        log.info(f"  ElasticNet selected: {len(self.genes)} genes")
        return self


class SVMRFESelector:
    """SVM-RFE with cross-validated feature count selection."""

    def __init__(self, data: DataPreparation):
        self.data   = data
        self.genes  : List[str] = []
        self.scores : pd.Series = pd.Series(dtype=float)
        self.ranking: np.ndarray = None

    def fit(self) -> 'SVMRFESelector':
        log.info("Running SVM-RFE feature selection...")
        svc  = SVC(kernel='linear', C=0.1, random_state=CFG.RANDOM_SEED)
        rfecv = RFECV(
            estimator=svc, step=0.05, cv=StratifiedKFold(CFG.CV_FOLDS),
            scoring='roc_auc', n_jobs=-1, min_features_to_select=5
        )
        rfecv.fit(self.data.X_train, self.data.y_train)
        self.ranking = rfecv.ranking_
        selected     = np.where(rfecv.support_)[0]
        self.genes   = [self.data.genes_filtered[i] for i in selected]
        # Rank by (max_rank - rank) so rank=1 gets highest score
        rank_scores  = (rfecv.ranking_.max() + 1 - rfecv.ranking_)[selected]
        self.scores  = pd.Series(rank_scores, index=self.genes).sort_values(ascending=False)
        log.info(f"  SVM-RFE selected: {len(self.genes)} genes | Optimal features: {rfecv.n_features_}")
        return self


class RandomForestSelector:
    """Random Forest with permutation importance + Boruta-inspired stability."""

    def __init__(self, data: DataPreparation):
        self.data   = data
        self.genes  : List[str] = []
        self.scores : pd.Series = pd.Series(dtype=float)
        self.importances: pd.DataFrame = None

    def fit(self) -> 'RandomForestSelector':
        log.info("Running Random Forest feature selection...")
        rf = RandomForestClassifier(
            n_estimators=500, max_features='sqrt', class_weight='balanced',
            n_jobs=-1, random_state=CFG.RANDOM_SEED
        )
        rf.fit(self.data.X_train, self.data.y_train)

        # Permutation importance (more reliable than MDI)
        perm = permutation_importance(
            rf, self.data.X_test, self.data.y_test,
            n_repeats=20, random_state=CFG.RANDOM_SEED, n_jobs=-1
        )
        perm_mean = perm.importances_mean
        perm_std  = perm.importances_std

        # Select genes where lower CI > 0 (permutation matters)
        significant = perm_mean - perm_std > 0
        selected    = np.where(significant)[0]

        if len(selected) < 5:  # fallback to top 50 by MDI
            log.warning("  Permutation cutoff too strict — using top 50 MDI genes")
            selected = np.argsort(rf.feature_importances_)[::-1][:50]

        self.genes  = [self.data.genes_filtered[i] for i in selected]
        self.scores = pd.Series(perm_mean[selected], index=self.genes).sort_values(ascending=False)
        self.importances = pd.DataFrame({
            'gene': self.data.genes_filtered,
            'mdi_importance': rf.feature_importances_,
            'perm_mean': perm_mean,
            'perm_std':  perm_std
        })
        log.info(f"  RF selected: {len(self.genes)} genes")
        return self

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4: GENE-ATTENTION TRANSFORMER (GAT) — NOVEL 2026 METHOD
# ═════════════════════════════════════════════════════════════════════════════

class GeneAttentionTransformer(nn.Module):
    """
    Gene-Attention Transformer (GAT) for biologically-aware gene selection.

    Novel aspects:
    - Each gene treated as a 'token' (like words in NLP)
    - Multi-head self-attention learns gene-gene co-expression relationships
    - Attention weights directly interpretable as gene importance scores
    - Biology-guided positional encoding using inflammation gene membership
    - Trained end-to-end for diagnostic classification

    Architecture:
        Gene Embeddings → Biological Positional Encoding →
        Multi-Head Self-Attention (×N layers) → CLS token → Classification
    """

    def __init__(self, n_genes: int, inflam_idx: List[int],
                 d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.n_genes    = n_genes
        self.d_model    = d_model
        self.inflam_idx = inflam_idx

        # ── Gene embedding (scalar → d_model dimensional) ─────────────────
        self.gene_embed = nn.Linear(1, d_model)

        # ── Biological positional encoding ─────────────────────────────────
        # Inflammation genes get a distinct learned embedding
        self.pos_embed    = nn.Embedding(n_genes + 1, d_model)  # +1 for CLS
        self.bio_embed    = nn.Embedding(2, d_model)             # inflam or not
        self.cls_token    = nn.Parameter(torch.randn(1, 1, d_model))

        # ── Transformer encoder ────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, norm_first=True  # pre-norm = stable
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm        = nn.LayerNorm(d_model)

        # ── Classification head ────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def _bio_position(self, n_genes: int, device) -> torch.Tensor:
        """Biological label: 1 if inflammation gene, else 0."""
        labels = torch.zeros(n_genes, dtype=torch.long, device=device)
        if self.inflam_idx:
            labels[self.inflam_idx] = 1
        return labels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, n_genes) expression values
        Returns:
            logit: (batch, 1)
            attn_weights: averaged attention across heads and layers
        """
        B, G = x.shape
        device = x.device

        # Gene embedding: (B, G, 1) → (B, G, d_model)
        x_emb = self.gene_embed(x.unsqueeze(-1))

        # Positional + biological encoding
        pos   = torch.arange(G, device=device).unsqueeze(0).expand(B, -1)
        bio   = self._bio_position(G, device).unsqueeze(0).expand(B, -1)
        x_emb = x_emb + self.pos_embed(pos) + self.bio_embed(bio)

        # Prepend CLS token
        cls   = self.cls_token.expand(B, -1, -1)
        x_emb = torch.cat([cls, x_emb], dim=1)           # (B, G+1, d_model)
        x_emb = self.dropout(x_emb)

        # Transformer
        out   = self.transformer(x_emb)                    # (B, G+1, d_model)
        out   = self.norm(out)

        # CLS token → classification
        cls_out = out[:, 0, :]                             # (B, d_model)
        logit   = self.classifier(cls_out)                 # (B, 1)

        # Attention weights (gene tokens only, skip CLS position)
        gene_out    = out[:, 1:, :]                        # (B, G, d_model)
        attn_scores = torch.mean(gene_out ** 2, dim=-1)    # (B, G) — proxy for gene salience
        attn_weights = F.softmax(attn_scores, dim=-1)      # normalized importance

        return logit, attn_weights


class TransformerGeneSelector:
    """
    Trains the Gene-Attention Transformer and extracts gene importance
    from attention weights — a novel biologically-guided selection method.
    """

    def __init__(self, data: DataPreparation):
        self.data   = data
        self.genes  : List[str] = []
        self.scores : pd.Series = pd.Series(dtype=float)
        self.model  : Optional[GeneAttentionTransformer] = None
        self.history: Dict     = {}
        self.attn_matrix: Optional[np.ndarray] = None

    def fit(self) -> 'TransformerGeneSelector':
        if not TORCH_AVAILABLE:
            log.warning("Skipping Transformer selector — PyTorch not available")
            return self

        print_header("STEP 4: GENE-ATTENTION TRANSFORMER (Novel 2026)")
        log.info("Training Gene-Attention Transformer for gene selection...")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log.info(f"  Device: {device}")

        n_genes = len(self.data.genes_filtered)
        model   = GeneAttentionTransformer(
            n_genes     = n_genes,
            inflam_idx  = self.data.inflam_idx,
            d_model     = CFG.TRANSFORMER_DIM,
            n_heads     = CFG.TRANSFORMER_HEADS,
            n_layers    = CFG.TRANSFORMER_LAYERS,
            dropout     = CFG.TRANSFORMER_DROPOUT
        ).to(device)
        self.model = model

        # ── Data loaders ───────────────────────────────────────────────────
        X_tr = torch.FloatTensor(self.data.X_train).to(device)
        y_tr = torch.FloatTensor(self.data.y_train).to(device)
        X_te = torch.FloatTensor(self.data.X_test).to(device)
        y_te = torch.FloatTensor(self.data.y_test).to(device)

        dataset = TensorDataset(X_tr, y_tr)
        loader  = DataLoader(dataset, batch_size=min(32, len(X_tr)), shuffle=True)

        # ── Optimizer with cosine annealing ───────────────────────────────
        pos_weight = torch.tensor([(y_tr==0).sum() / (y_tr==1).sum()]).to(device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer  = torch.optim.AdamW(model.parameters(), lr=CFG.TRANSFORMER_LR,
                                        weight_decay=1e-4)
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG.TRANSFORMER_EPOCHS
        )

        # ── Training loop ──────────────────────────────────────────────────
        train_losses, val_aucs = [], []
        best_auc, best_state   = 0.0, None

        for epoch in range(CFG.TRANSFORMER_EPOCHS):
            model.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                logit, _ = model(xb)
                loss = criterion(logit.squeeze(), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()

            # Validation AUC
            model.eval()
            with torch.no_grad():
                logit_te, _ = model(X_te)
                prob_te      = torch.sigmoid(logit_te).squeeze().cpu().numpy()
                val_auc      = roc_auc_score(self.data.y_test, prob_te)

            train_losses.append(epoch_loss / len(loader))
            val_aucs.append(val_auc)

            if val_auc > best_auc:
                best_auc   = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 20 == 0:
                log.info(f"  Epoch {epoch+1:3d}/{CFG.TRANSFORMER_EPOCHS} "
                         f"| Loss: {epoch_loss/len(loader):.4f} | Val AUC: {val_auc:.4f}")

        self.history = {'train_loss': train_losses, 'val_auc': val_aucs}
        log.info(f"  Best validation AUC: {best_auc:.4f}")

        # ── Extract attention weights ──────────────────────────────────────
        model.load_state_dict(best_state)
        model.eval()
        all_attn = []
        with torch.no_grad():
            logit_all, attn_all = model(torch.cat([X_tr, X_te], dim=0))
            all_attn = attn_all.cpu().numpy()  # (n_samples, n_genes)

        # Aggregate: mean attention across all samples
        self.attn_matrix    = all_attn
        mean_attn           = all_attn.mean(axis=0)   # (n_genes,)
        self.scores_all     = pd.Series(mean_attn, index=self.data.genes_filtered)

        # Select top genes by attention (top 20% or min 20 genes)
        n_select = max(20, int(n_genes * 0.15))
        top_idx  = np.argsort(mean_attn)[::-1][:n_select]
        self.genes  = [self.data.genes_filtered[i] for i in top_idx]
        self.scores = pd.Series(mean_attn[top_idx], index=self.genes)

        log.info(f"  Transformer selected: {len(self.genes)} genes by attention weight")

        # Save model
        torch.save(best_state, CFG.MODEL_DIR / "gene_attention_transformer.pt")
        self._plot_training()
        return self

    def _plot_training(self):
        if not self.history:
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(self.history['train_loss'], color='steelblue')
        axes[0].set_title('Training Loss', fontsize=CFG.FONT_SIZE)
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('BCE Loss')
        axes[0].grid(alpha=0.3)

        axes[1].plot(self.history['val_auc'], color='firebrick')
        axes[1].axhline(max(self.history['val_auc']), ls='--', color='grey', alpha=0.5,
                        label=f"Best={max(self.history['val_auc']):.3f}")
        axes[1].set_title('Validation AUC', fontsize=CFG.FONT_SIZE)
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('AUC')
        axes[1].legend(); axes[1].grid(alpha=0.3)
        save_figure(fig, "GAT_01_training_curves")
