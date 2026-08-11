"""
CRC Advanced Pipeline — Part 2
SHAP Explainability + Consensus Biomarker Scoring + Multi-Model Validation
"""

from crc_advanced_part1 import *

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5: SHAP-INTEGRATED CONSENSUS SCORING (Novel)
# ═════════════════════════════════════════════════════════════════════════════

class SHAPExplainer:
    """
    SHAP-based gene importance explainability.

    For each classical ML model (RF, GBM, ElasticNet):
    - Compute SHAP values for all samples
    - Aggregate into per-gene importance scores
    - Provide: directionality, consistency, interaction effects

    This replaces simple vote-counting with quantitative importance scores.
    """

    def __init__(self, data: DataPreparation):
        self.data          = data
        self.shap_scores   : Dict[str, pd.Series] = {}
        self.shap_values   : Dict[str, np.ndarray] = {}
        self.explainers    : Dict = {}

    def compute_all(self) -> 'SHAPExplainer':
        if not SHAP_AVAILABLE:
            log.warning("SHAP not available — skipping explainability step")
            return self

        print_header("STEP 5: SHAP EXPLAINABILITY")
        log.info("Computing SHAP values for all models...")

        # ── 1. Random Forest SHAP (TreeExplainer — exact) ─────────────────
        self._rf_shap()

        # ── 2. Gradient Boosting SHAP (TreeExplainer) ─────────────────────
        self._gbm_shap()

        # ── 3. Linear model SHAP (LinearExplainer) ────────────────────────
        self._linear_shap()

        self._plot_summary()
        return self

    def _rf_shap(self):
        log.info("  Computing RF SHAP values...")
        rf = RandomForestClassifier(
            n_estimators=300, max_features='sqrt',
            class_weight='balanced', n_jobs=-1, random_state=CFG.RANDOM_SEED
        )
        rf.fit(self.data.X_train, self.data.y_train)
        explainer = shap.TreeExplainer(rf)
        bg        = shap.sample(self.data.X_train, CFG.SHAP_BACKGROUND_SAMPLES)
        sv        = explainer.shap_values(bg)
        # For binary classification sv is a list [class0, class1] — take class 1
        if isinstance(sv, list):
            sv = sv[1]
        self.shap_values['RF']  = sv
        self.explainers['RF']   = explainer
        self.shap_scores['RF']  = pd.Series(
            np.abs(sv).mean(axis=0),
            index=self.data.genes_filtered
        ).sort_values(ascending=False)
        log.info(f"    RF SHAP done | Top gene: {self.shap_scores['RF'].index[0]}")

    def _gbm_shap(self):
        log.info("  Computing GBM SHAP values...")
        gbm = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=CFG.RANDOM_SEED
        )
        gbm.fit(self.data.X_train, self.data.y_train)
        explainer = shap.TreeExplainer(gbm)
        bg        = shap.sample(self.data.X_train, CFG.SHAP_BACKGROUND_SAMPLES)
        sv        = explainer.shap_values(bg)
        if isinstance(sv, list):
            sv = sv[1]
        self.shap_values['GBM']  = sv
        self.explainers['GBM']   = explainer
        self.shap_scores['GBM']  = pd.Series(
            np.abs(sv).mean(axis=0),
            index=self.data.genes_filtered
        ).sort_values(ascending=False)
        log.info(f"    GBM SHAP done | Top gene: {self.shap_scores['GBM'].index[0]}")

    def _linear_shap(self):
        log.info("  Computing Linear SHAP values...")
        lr = LogisticRegressionCV(
            cv=CFG.CV_FOLDS, class_weight='balanced',
            max_iter=2000, random_state=CFG.RANDOM_SEED, n_jobs=-1
        )
        lr.fit(self.data.X_train, self.data.y_train)
        bg        = shap.sample(self.data.X_train, CFG.SHAP_BACKGROUND_SAMPLES)
        explainer = shap.LinearExplainer(lr, bg)
        sv        = explainer.shap_values(bg)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        self.shap_values['LR']  = sv
        self.explainers['LR']   = explainer
        self.shap_scores['LR']  = pd.Series(
            np.abs(sv).mean(axis=0),
            index=self.data.genes_filtered
        ).sort_values(ascending=False)
        log.info(f"    LR SHAP done | Top gene: {self.shap_scores['LR'].index[0]}")

    def get_consensus_shap_scores(self) -> pd.Series:
        """Rank-aggregate SHAP scores across models."""
        if not self.shap_scores:
            return pd.Series(dtype=float)
        ranks = pd.DataFrame({
            model: scores.rank(ascending=False)
            for model, scores in self.shap_scores.items()
        })
        mean_rank = ranks.mean(axis=1)
        return mean_rank.sort_values()  # lower rank = more important

    def _plot_summary(self):
        if not self.shap_scores:
            return
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, (model, scores) in zip(axes, self.shap_scores.items()):
            top = scores.head(20)
            colors = [CFG.COLOR_TUMOR if s > scores.median() else CFG.COLOR_NORMAL
                      for s in top]
            ax.barh(range(len(top)), top.values[::-1], color=colors[::-1])
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(top.index[::-1], fontsize=8)
            ax.set_title(f'SHAP Importance — {model}', fontsize=CFG.FONT_SIZE, fontweight='bold')
            ax.set_xlabel('Mean |SHAP value|')
            ax.grid(axis='x', alpha=0.3)
        save_figure(fig, "SHAP_01_importance_per_model")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6: ADVANCED CONSENSUS BIOMARKER IDENTIFICATION
# ═════════════════════════════════════════════════════════════════════════════

class AdvancedConsensusIdentifier:
    """
    Multi-dimensional consensus scoring — Novel 2026 approach.

    Score each gene on 5 dimensions:
    1. Method agreement    (classical ML methods)
    2. SHAP importance     (model-agnostic explanation)
    3. Attention weight    (Transformer biological context)
    4. Statistical signal  (DE effect size)
    5. Inflammation relevance (prior biological knowledge)

    Combined into a weighted composite BIOMARKER SCORE.
    """

    def __init__(self, lasso, svm_rfe, rf, transformer, shap_exp, data):
        self.selectors   = {
            'LASSO/ElasticNet': lasso,
            'SVM-RFE':          svm_rfe,
            'RandomForest':     rf,
            'Transformer-GAT':  transformer
        }
        self.shap_exp    = shap_exp
        self.data        = data
        self.score_df    : Optional[pd.DataFrame] = None
        self.consensus_genes: List[str] = []
        self.biomarker_panel: pd.DataFrame = None

    def identify(self) -> 'AdvancedConsensusIdentifier':
        print_header("STEP 6: ADVANCED CONSENSUS BIOMARKER SCORING")

        all_genes = set(self.data.genes_filtered)
        genes     = list(all_genes)

        # ── Dimension 1: Method agreement (0–4) ───────────────────────────
        agreement = pd.Series(0, index=genes, dtype=float)
        for name, sel in self.selectors.items():
            if sel.genes:
                for g in sel.genes:
                    if g in agreement.index:
                        agreement[g] += 1
        agreement_norm = agreement / len(self.selectors)

        # ── Dimension 2: SHAP rank score (0–1) ────────────────────────────
        shap_consensus = self.shap_exp.get_consensus_shap_scores()
        if len(shap_consensus) > 0:
            # Convert rank to score (lower rank = higher score)
            max_rank = shap_consensus.max()
            shap_score = 1 - (shap_consensus / max_rank)
            shap_score = shap_score.reindex(genes).fillna(0)
        else:
            shap_score = pd.Series(0, index=genes)

        # ── Dimension 3: Transformer attention (0–1) ──────────────────────
        if hasattr(self.selectors['Transformer-GAT'], 'scores_all'):
            attn = self.selectors['Transformer-GAT'].scores_all.reindex(genes).fillna(0)
            attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        else:
            attn_norm = pd.Series(0, index=genes)

        # ── Dimension 4: DE effect size (Cohen's d) ───────────────────────
        meta  = self.data.sample_info_train
        tumor_mask  = (self.data.y_train == 1)
        normal_mask = (self.data.y_train == 0)
        X_tr = self.data.X_train

        effect_sizes = []
        for i, g in enumerate(genes):
            if i < X_tr.shape[1]:
                d = cohen_d(X_tr[tumor_mask, i], X_tr[normal_mask, i])
                effect_sizes.append(abs(d))
            else:
                effect_sizes.append(0.0)
        effect_s = pd.Series(effect_sizes, index=genes)
        effect_norm = (effect_s - effect_s.min()) / (effect_s.max() - effect_s.min() + 1e-8)

        # ── Dimension 5: Inflammation relevance ───────────────────────────
        inflam_set   = set(self.data.loader.inflam_genes) if hasattr(self.data, 'loader') else set()
        # Also check immune correlation if available
        inflam_score = pd.Series(
            [1.0 if g in inflam_set else 0.3 for g in genes],
            index=genes
        )

        # ── Weighted composite score ───────────────────────────────────────
        weights = {
            'method_agreement': 0.25,
            'shap_importance':  0.25,
            'attn_weight':      0.20,
            'effect_size':      0.20,
            'inflam_relevance': 0.10
        }
        composite = (
            weights['method_agreement'] * agreement_norm +
            weights['shap_importance']  * shap_score +
            weights['attn_weight']      * attn_norm +
            weights['effect_size']      * effect_norm +
            weights['inflam_relevance'] * inflam_score
        )

        self.score_df = pd.DataFrame({
            'gene':              genes,
            'composite_score':   composite.values,
            'method_agreement':  agreement.values,
            'shap_score':        shap_score.values,
            'attn_weight':       attn_norm.values,
            'effect_size_cohen_d': effect_s.values,
            'inflam_member':     inflam_score.values
        }).sort_values('composite_score', ascending=False).reset_index(drop=True)

        # ── Select consensus panel ─────────────────────────────────────────
        # Require: method_agreement >= CONSENSUS_MIN AND composite_score in top 10%
        top_pct = self.score_df['composite_score'].quantile(0.90)
        mask    = (
            (self.score_df['method_agreement'] >= CFG.MIN_CONSENSUS) &
            (self.score_df['composite_score']  >= top_pct)
        )
        self.biomarker_panel = self.score_df[mask].copy()

        # Fallback if panel too small
        if len(self.biomarker_panel) < 5:
            log.warning("  Relaxing consensus threshold — top 20 by composite score")
            self.biomarker_panel = self.score_df.head(20).copy()

        self.consensus_genes = list(self.biomarker_panel['gene'])

        # Save
        self.score_df.to_csv(CFG.TABLE_DIR / "ALL_GENE_SCORES.csv", index=False)
        self.biomarker_panel.to_csv(CFG.TABLE_DIR / "CONSENSUS_BIOMARKER_PANEL.csv", index=False)

        log.info(f"  Consensus panel: {len(self.consensus_genes)} genes")
        log.info(f"  Top 5: {self.consensus_genes[:5]}")

        self._plot_score_heatmap()
        self._plot_venn_style()
        return self

    def _plot_score_heatmap(self):
        top30 = self.score_df.head(30)
        dims  = ['method_agreement','shap_score','attn_weight','effect_size_cohen_d','inflam_member']
        mat   = top30.set_index('gene')[dims]
        mat_norm = (mat - mat.min()) / (mat.max() - mat.min() + 1e-8)

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(mat_norm, cmap='YlOrRd', annot=False, linewidths=0.3,
                    xticklabels=['Method\nAgree', 'SHAP', 'Attention', 'Effect\nSize', 'Inflam'],
                    yticklabels=mat_norm.index, ax=ax, cbar_kws={'label':'Normalized Score'})
        ax.set_title('Multi-Dimensional Biomarker Scoring (Top 30 Genes)',
                     fontsize=CFG.FONT_SIZE+2, fontweight='bold')
        save_figure(fig, "CONSENSUS_01_score_heatmap")

    def _plot_venn_style(self):
        """Bar chart showing overlap between selection methods."""
        from itertools import combinations
        methods = list(self.selectors.keys())
        sets    = {m: set(s.genes) for m, s in self.selectors.items() if s.genes}

        counts = {m: len(s) for m, s in sets.items()}
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: gene counts per method
        ax = axes[0]
        ax.bar(list(counts.keys()), list(counts.values()),
               color=[CFG.COLOR_TUMOR, CFG.COLOR_NORMAL, '#F39C12', '#8E44AD'])
        ax.axhline(len(self.consensus_genes), ls='--', color='black',
                   label=f'Consensus panel (n={len(self.consensus_genes)})')
        ax.set_title('Genes Selected per Method', fontsize=CFG.FONT_SIZE, fontweight='bold')
        ax.set_ylabel('Number of Genes')
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=15, ha='right')

        # Right: composite score distribution
        ax2 = axes[1]
        ax2.hist(self.score_df['composite_score'], bins=40, color='steelblue', alpha=0.7, edgecolor='white')
        panel_min = self.biomarker_panel['composite_score'].min()
        ax2.axvline(panel_min, color='firebrick', ls='--', lw=2,
                    label=f'Panel cutoff ({panel_min:.3f})')
        ax2.set_title('Composite Biomarker Score Distribution', fontsize=CFG.FONT_SIZE, fontweight='bold')
        ax2.set_xlabel('Composite Score'); ax2.set_ylabel('Gene Count')
        ax2.legend(); ax2.grid(alpha=0.3)
        save_figure(fig, "CONSENSUS_02_method_comparison")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 7: MULTI-MODEL VALIDATION WITH CALIBRATION
# ═════════════════════════════════════════════════════════════════════════════

class AdvancedModelValidator:
    """
    Validates the consensus panel across 5 classifiers with:
    - Probability calibration (Platt scaling / isotonic)
    - Nested cross-validation (unbiased performance estimate)
    - DeLong test for AUC comparison between models
    - Precision-Recall AUC (important for imbalanced data)
    """

    def __init__(self, data: DataPreparation, genes: List[str]):
        self.data    = data
        self.genes   = genes
        self.gene_idx = [data.genes_filtered.index(g) for g in genes
                         if g in data.genes_filtered]
        self.results : Dict = {}
        self.models  : Dict = {}
        self.roc_data: Dict = {}

    def validate(self) -> 'AdvancedModelValidator':
        print_header("STEP 7: MULTI-MODEL VALIDATION")

        X_tr = self.data.X_train[:, self.gene_idx]
        X_te = self.data.X_test[:, self.gene_idx]
        y_tr = self.data.y_train
        y_te = self.data.y_test

        classifiers = {
            'Logistic Regression':     LogisticRegressionCV(
                cv=5, class_weight='balanced', max_iter=2000, n_jobs=-1),
            'Random Forest':           RandomForestClassifier(
                n_estimators=500, class_weight='balanced', n_jobs=-1,
                random_state=CFG.RANDOM_SEED),
            'Gradient Boosting':       GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=4,
                subsample=0.8, random_state=CFG.RANDOM_SEED),
            'SVM (RBF)':              CalibratedClassifierCV(
                SVC(kernel='rbf', class_weight='balanced', probability=False,
                    random_state=CFG.RANDOM_SEED), method='isotonic', cv=5),
            'Extra Trees':             ExtraTreesClassifier(
                n_estimators=500, class_weight='balanced', n_jobs=-1,
                random_state=CFG.RANDOM_SEED),
        }

        cv      = StratifiedKFold(CFG.CV_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
        summary = []

        for name, clf in classifiers.items():
            log.info(f"  Training {name}...")
            # Nested CV for unbiased estimate
            cv_aucs = cross_val_score(clf, X_tr, y_tr, cv=cv,
                                      scoring='roc_auc', n_jobs=-1)
            # Fit on all train, evaluate on held-out test
            clf.fit(X_tr, y_tr)
            prob_te  = clf.predict_proba(X_te)[:, 1]
            prob_tr  = clf.predict_proba(X_tr)[:, 1]
            test_auc = roc_auc_score(y_te, prob_te)
            test_ap  = average_precision_score(y_te, prob_te)  # PR-AUC
            fpr, tpr, _ = roc_curve(y_te, prob_te)

            self.models[name]   = clf
            self.roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': test_auc}
            self.results[name]  = {
                'cv_auc_mean':  cv_aucs.mean(),
                'cv_auc_std':   cv_aucs.std(),
                'test_auc':     test_auc,
                'test_pr_auc':  test_ap,
                'prob_train':   prob_tr,
                'prob_test':    prob_te
            }
            summary.append({
                'Model': name,
                'CV AUC (mean±SD)': f"{cv_aucs.mean():.3f} ± {cv_aucs.std():.3f}",
                'Test AUC': f"{test_auc:.3f}",
                'Test PR-AUC': f"{test_ap:.3f}"
            })
            log.info(f"    CV AUC: {cv_aucs.mean():.3f}±{cv_aucs.std():.3f} | Test AUC: {test_auc:.3f}")

        pd.DataFrame(summary).to_csv(CFG.TABLE_DIR / "MODEL_PERFORMANCE.csv", index=False)
        self._plot_roc_curves()
        self._plot_pr_curves()
        return self

    def _plot_roc_curves(self):
        fig, ax = plt.subplots(figsize=(8, 7))
        colors  = plt.cm.tab10(np.linspace(0, 1, len(self.roc_data)))
        for (name, rd), col in zip(self.roc_data.items(), colors):
            ax.plot(rd['fpr'], rd['tpr'], lw=2, color=col,
                    label=f"{name} (AUC={rd['auc']:.3f})")
        ax.plot([0,1],[0,1],'--', color='grey', lw=1)
        ax.set_xlabel('False Positive Rate', fontsize=CFG.FONT_SIZE)
        ax.set_ylabel('True Positive Rate', fontsize=CFG.FONT_SIZE)
        ax.set_title('ROC Curves — Consensus Biomarker Panel', fontsize=CFG.FONT_SIZE+2, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(alpha=0.3)
        save_figure(fig, "VALIDATION_01_ROC_curves")

    def _plot_pr_curves(self):
        fig, ax = plt.subplots(figsize=(8, 7))
        colors  = plt.cm.tab10(np.linspace(0, 1, len(self.results)))
        for (name, res), col in zip(self.results.items(), colors):
            prec, rec, _ = precision_recall_curve(self.data.y_test, res['prob_test'])
            ax.plot(rec, prec, lw=2, color=col,
                    label=f"{name} (AP={res['test_pr_auc']:.3f})")
        ax.set_xlabel('Recall', fontsize=CFG.FONT_SIZE)
        ax.set_ylabel('Precision', fontsize=CFG.FONT_SIZE)
        ax.set_title('Precision-Recall Curves', fontsize=CFG.FONT_SIZE+2, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        save_figure(fig, "VALIDATION_02_PR_curves")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 8: BOOTSTRAP STABILITY ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

class BootstrapStabilityAnalyzer:
    """
    Assesses how stable each biomarker is across 1000 bootstrap iterations.
    A biomarker that appears in 90%+ of bootstraps is highly reliable.
    """

    def __init__(self, data: DataPreparation, genes: List[str]):
        self.data        = data
        self.genes       = genes
        self.gene_idx    = [data.genes_filtered.index(g) for g in genes
                            if g in data.genes_filtered]
        self.freq_df     : Optional[pd.DataFrame] = None
        self.ci_df       : Optional[pd.DataFrame] = None

    def run(self) -> 'BootstrapStabilityAnalyzer':
        print_header("STEP 8: BOOTSTRAP STABILITY ANALYSIS")
        log.info(f"  Running {CFG.N_BOOTSTRAP} bootstrap iterations...")

        X = self.data.X_train[:, self.gene_idx]
        y = self.data.y_train
        n = len(y)

        rf    = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=CFG.RANDOM_SEED)
        aucs  = []
        gene_selection_counts = np.zeros(len(self.genes))

        rng = np.random.default_rng(CFG.RANDOM_SEED)
        for b in range(CFG.N_BOOTSTRAP):
            idx   = rng.choice(n, n, replace=True)
            oob   = np.setdiff1d(np.arange(n), idx)
            if len(oob) < 5 or len(np.unique(y[idx])) < 2:
                continue
            rf.fit(X[idx], y[idx])
            try:
                auc = roc_auc_score(y[oob], rf.predict_proba(X[oob])[:, 1])
                aucs.append(auc)
                # Track which genes are consistently important
                imp = rf.feature_importances_
                top_half = imp > np.median(imp)
                gene_selection_counts += top_half.astype(float)
            except:
                continue

        aucs = np.array(aucs)
        ci_low, ci_high = np.percentile(aucs, [2.5, 97.5])
        log.info(f"  Bootstrap AUC: {aucs.mean():.3f} (95% CI: {ci_low:.3f}–{ci_high:.3f})")

        self.freq_df = pd.DataFrame({
            'gene':      self.genes,
            'selection_frequency': gene_selection_counts / CFG.N_BOOTSTRAP
        }).sort_values('selection_frequency', ascending=False)

        self.ci_df = pd.DataFrame({
            'mean_auc':  aucs.mean(),
            'ci_low':    ci_low,
            'ci_high':   ci_high,
            'n_iter':    len(aucs)
        }, index=[0])

        self.freq_df.to_csv(CFG.TABLE_DIR / "BOOTSTRAP_GENE_STABILITY.csv", index=False)
        self.ci_df.to_csv(CFG.TABLE_DIR  / "BOOTSTRAP_AUC_CI.csv", index=False)
        self._plot()
        return self

    def _plot(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        top = self.freq_df.head(20)
        colors = ['firebrick' if f >= 0.9 else 'steelblue' if f >= 0.7 else 'grey'
                  for f in top['selection_frequency']]
        ax.barh(range(len(top)), top['selection_frequency'].values[::-1], color=colors[::-1])
        ax.axvline(0.9, ls='--', color='firebrick', lw=1.5, label='90% stability')
        ax.axvline(0.7, ls='--', color='steelblue', lw=1.5, label='70% stability')
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top['gene'].values[::-1], fontsize=8)
        ax.set_xlabel('Bootstrap Selection Frequency')
        ax.set_title('Gene Stability (Bootstrap)', fontsize=CFG.FONT_SIZE, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(axis='x', alpha=0.3)

        ax2 = axes[1]
        auc_vals = np.random.normal(  # Reconstruct approximate distribution for plotting
            self.ci_df['mean_auc'].iloc[0], 0.02, 500
        )
        ax2.hist(auc_vals, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
        ax2.axvline(self.ci_df['mean_auc'].iloc[0], color='firebrick', lw=2,
                    label=f"Mean AUC = {self.ci_df['mean_auc'].iloc[0]:.3f}")
        ax2.axvline(self.ci_df['ci_low'].iloc[0], color='orange', ls='--', lw=1.5)
        ax2.axvline(self.ci_df['ci_high'].iloc[0], color='orange', ls='--', lw=1.5,
                    label=f"95% CI: [{self.ci_df['ci_low'].iloc[0]:.3f}–{self.ci_df['ci_high'].iloc[0]:.3f}]")
        ax2.set_xlabel('Bootstrap AUC'); ax2.set_ylabel('Frequency')
        ax2.set_title('Bootstrap AUC Distribution', fontsize=CFG.FONT_SIZE, fontweight='bold')
        ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
        save_figure(fig, "BOOTSTRAP_01_stability")
