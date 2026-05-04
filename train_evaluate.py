#!/usr/bin/env python3
"""CMPE188 HW3: Nemotron `task_type` text classification.

Builds sklearn Pipelines over TF-IDF, Word2Vec mean embeddings, and several
classifiers; evaluates hold-out metrics, stratified CV, train/inference timing,
and saves plots plus CSV/JSON under --output-dir (default: figures/).
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

DEFAULT_DATA_URL = "https://raw.githubusercontent.com/lkk688/CoderGym/main/Nemotron/train_with_task_type.csv"


def download_if_needed(path: Path, url: str = DEFAULT_DATA_URL) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset to {path} ...")
    urllib.request.urlretrieve(url, path)


def extract_query_tail(text: str) -> str:
    """Final instruction line only — drops shared 'Alice' framing and worked examples."""
    lines = [
        ln.strip()
        for ln in str(text).splitlines()
        if ln.strip().lower().startswith("now,")
    ]
    if lines:
        return lines[-1]
    for ln in reversed(str(text).splitlines()):
        s = ln.strip()
        if s:
            return s
    return str(text).strip()


def extract_payload(text: str) -> str:
    """Strip instruction phrasing; classify from the final asked item (after last ':')."""
    tail = extract_query_tail(text)
    if ":" in tail:
        return tail.rsplit(":", 1)[-1].strip()
    return tail


def load_dataset(csv_path: Path, text_mode: str = "prompt_payload"):
    """Load texts and string labels; ``text_mode`` selects how ``prompt`` is derived."""
    df = pd.read_csv(csv_path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["prompt", "task_type"])
    df["task_type"] = df["task_type"].str.strip()
    if text_mode == "prompt":
        texts = df["prompt"].fillna("")
    elif text_mode == "prompt_answer":
        ans = df.get("answer", pd.Series([""] * len(df)))
        texts = df["prompt"].fillna("") + " " + ans.fillna("")
    elif text_mode == "prompt_tail":
        texts = df["prompt"].fillna("").map(extract_query_tail)
    elif text_mode == "prompt_payload":
        texts = df["prompt"].fillna("").map(extract_payload)
    else:
        raise ValueError(
            "text_mode must be prompt, prompt_answer, prompt_tail, or prompt_payload"
        )
    return texts, df["task_type"]


_word_re = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize_simple(text: str):
    return _word_re.findall(text.lower())


class MeanWord2VecVectorizer(BaseEstimator, TransformerMixin):
    """Fit gensim Word2Vec on training tokens; transform = mean of word vectors."""

    def __init__(
        self,
        vector_size: int = 128,
        window: int = 5,
        min_count: int = 2,
        workers: int = 1 if sys.platform == "win32" else 4,
        epochs: int = 15,
        seed: int = 42,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.seed = seed

    def fit(self, X: np.ndarray, y=None):
        sents = [tokenize_simple(str(t)) for t in X.ravel()]
        self.w2v_ = Word2Vec(
            sentences=sents,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            seed=self.seed,
            sg=0,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        model = self.w2v_
        dim = model.vector_size
        out = np.zeros((len(X), dim), dtype=np.float32)
        for i, raw in enumerate(X.ravel()):
            toks = tokenize_simple(str(raw))
            got = [model.wv[w] for w in toks if w in model.wv]
            if got:
                out[i] = np.mean(got, axis=0)
        return out


def _new_tfidf():
    return TfidfVectorizer(
        max_features=30_000,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )


def build_pipelines(random_state: int) -> dict[str, Pipeline]:
    """All baseline pipelines for TF-IDF and Word2Vec representations."""
    out: dict[str, Pipeline] = {}

    # MultinomialNB: sklearn expects nonnegative inputs; TF-IDF weights are >= 0.
    out["TF-IDF + Multinomial NB"] = Pipeline(
        [("tfidf", _new_tfidf()), ("clf", MultinomialNB(alpha=0.1))]
    )

    out["TF-IDF + Linear SVM"] = Pipeline(
        [
            ("tfidf", _new_tfidf()),
            ("clf", LinearSVC(random_state=random_state, dual=False)),
        ]
    )

    out["TF-IDF + k-NN (cosine)"] = Pipeline(
        [
            ("tfidf", _new_tfidf()),
            (
                "clf",
                KNeighborsClassifier(
                    n_neighbors=7,
                    metric="cosine",
                    algorithm="brute",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    out["TF-IDF + Decision Tree"] = Pipeline(
        [
            ("tfidf", _new_tfidf()),
            ("clf", DecisionTreeClassifier(random_state=random_state, max_depth=40)),
        ]
    )

    out["TF-IDF + Random Forest"] = Pipeline(
        [
            ("tfidf", _new_tfidf()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=120,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    tfidf_mlp = TfidfVectorizer(
        max_features=25_000,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    out["TF-IDF + SVD + MLP (NN)"] = Pipeline(
        [
            ("tfidf", tfidf_mlp),
            ("svd", TruncatedSVD(n_components=300, random_state=random_state)),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(256, 128),
                    activation="relu",
                    max_iter=80,
                    random_state=random_state,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=8,
                ),
            ),
        ]
    )

    w2v_rf = MeanWord2VecVectorizer(vector_size=128, epochs=15, seed=random_state)
    out["Word2Vec + Random Forest"] = Pipeline(
        [
            ("w2v", w2v_rf),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=120, random_state=random_state, n_jobs=-1
                ),
            ),
        ]
    )

    w2v_vec = MeanWord2VecVectorizer(vector_size=128, epochs=15, seed=random_state)
    out["Word2Vec + k-NN"] = Pipeline(
        [
            ("w2v", w2v_vec),
            ("clf", KNeighborsClassifier(n_neighbors=9, metric="euclidean", n_jobs=-1)),
        ]
    )

    w2v_nb = MeanWord2VecVectorizer(vector_size=128, epochs=15, seed=random_state)
    out["Word2Vec + Gaussian NB"] = Pipeline([("w2v", w2v_nb), ("clf", GaussianNB())])

    w2v_mlp = MeanWord2VecVectorizer(vector_size=128, epochs=15, seed=random_state)
    out["Word2Vec + MLP (NN)"] = Pipeline(
        [
            ("w2v", w2v_mlp),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(256, 128),
                    max_iter=100,
                    random_state=random_state,
                    early_stopping=True,
                    validation_fraction=0.1,
                ),
            ),
        ]
    )

    return out


FORMAL_THREE = [
    # Assignment “three distinct pipelines”: two TF-IDF + one Word2Vec pipeline.
    "TF-IDF + Linear SVM",
    "TF-IDF + Multinomial NB",
    "Word2Vec + Random Forest",
]


def cross_validate_stratified(
    estimator: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    seed: int,
) -> dict[str, float]:
    """Out-of-fold metrics on stratified K folds (same preprocessing inside each clone)."""
    if n_splits < 2:
        return {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs: list[float] = []
    precs: list[float] = []
    recs: list[float] = []
    f1s: list[float] = []
    for tr_idx, va_idx in skf.split(X, y):
        est = clone(estimator)
        est.fit(X[tr_idx], y[tr_idx])
        pred = est.predict(X[va_idx])
        y_va = y[va_idx]
        accs.append(accuracy_score(y_va, pred))
        precs.append(precision_score(y_va, pred, average="macro", zero_division=0))
        recs.append(recall_score(y_va, pred, average="macro", zero_division=0))
        f1s.append(f1_score(y_va, pred, average="macro", zero_division=0))
    return {
        "cv_accuracy_mean": float(np.mean(accs)),
        "cv_accuracy_std": float(np.std(accs)),
        "cv_precision_macro_mean": float(np.mean(precs)),
        "cv_precision_macro_std": float(np.std(precs)),
        "cv_recall_macro_mean": float(np.mean(recs)),
        "cv_recall_macro_std": float(np.std(recs)),
        "cv_f1_macro_mean": float(np.mean(f1s)),
        "cv_f1_macro_std": float(np.std(f1s)),
    }


def evaluate_one(
    name, model, X_train, y_train, X_test, y_test, class_names: np.ndarray
):
    """Fit on train, predict on test; return metrics, timings, and predictions."""
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    # Full-batch inference time on the held-out test set (assignment requirement).
    t1 = time.perf_counter()
    y_pred = model.predict(X_test)
    infer_time = time.perf_counter() - t1

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=list(class_names),
        zero_division=0,
    )

    return {
        "pipeline": name,
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "train_time_sec": float(train_time),
        "inference_time_sec": float(infer_time),
        "inference_time_per_1k_sec": float(infer_time / max(len(X_test), 1) * 1000.0),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "classification_report": report,
        "y_pred": y_pred,
    }


def plot_formal_confusion(
    rows: list[dict[str, Any]],
    formal: list[str],
    y_test: np.ndarray,
    class_names: np.ndarray,
    out_dir: Path,
) -> None:
    by_name = {r["pipeline"]: r for r in rows}
    formal_rows = [by_name[n] for n in formal if n in by_name]
    if len(formal_rows) != len(formal):
        return
    fig, axes = plt.subplots(1, len(formal_rows), figsize=(6 * len(formal_rows), 5.2))
    if len(formal_rows) == 1:
        axes = np.array([axes])
    labels = np.arange(len(class_names))
    sns.set_theme(style="whitegrid", context="talk")
    for ax, row in zip(axes.ravel(), formal_rows):
        cm = confusion_matrix(
            y_test,
            row["y_pred"],
            labels=labels,
            normalize="true",
        )
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            xticklabels=list(class_names),
            yticklabels=list(class_names),
            ax=ax,
        )
        ax.set_title(row["pipeline"], fontsize=11)
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)
    fig.suptitle("Normalized confusion (held-out test)", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_formal_three_confusion_normalized.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cv_f1_formal(df: pd.DataFrame, formal: list[str], n_splits: int, out_dir: Path) -> None:
    if "cv_f1_macro_mean" not in df.columns or n_splits < 2:
        return
    formal_df = df[df["pipeline"].isin(formal)].copy()
    if formal_df.empty:
        return
    order_map = {n: i for i, n in enumerate(formal)}
    formal_df["_order"] = formal_df["pipeline"].map(order_map)
    formal_df = formal_df.sort_values("_order").drop(columns=["_order"])
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(formal_df))
    means = formal_df["cv_f1_macro_mean"].values
    stds = formal_df["cv_f1_macro_std"].values
    ax.bar(x, means, yerr=stds, capsize=8, color="#4c72b0", edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(formal_df["pipeline"], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro F1")
    ax.set_title(f"Stratified {n_splits}-fold CV — macro F1 (mean ± std across folds)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_formal_three_cv_macro_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(df: pd.DataFrame, formal: list[str], out_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    formal_df = df[df["pipeline"].isin(formal)].copy()
    formal_df["_order"] = formal_df["pipeline"].map({n: i for i, n in enumerate(formal)})
    formal_df = formal_df.sort_values("_order").drop(columns=["_order"])

    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    x = np.arange(len(formal_df))
    width = 0.2
    for i, m in enumerate(metrics):
        lab = m.replace("_macro", "").replace("_", " ")
        ax.bar(x + i * width, formal_df[m], width=width, label=lab)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(formal_df["pipeline"], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend(loc="lower right", fontsize="small")
    ax.set_title("Held-out test: macro metrics (formal 3)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_formal_three_metrics_bar.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    idx = np.arange(len(formal_df))
    w = 0.35
    ax.bar(idx - w / 2, formal_df["train_time_sec"], width=w, label="Train")
    ax.bar(idx + w / 2, formal_df["inference_time_sec"], width=w, label="Inference")
    ax.set_xticks(idx)
    ax.set_xticklabels(formal_df["pipeline"], rotation=15, ha="right")
    ax.set_ylabel("seconds")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Train vs inference time")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_formal_three_train_infer_bar.png", dpi=150)
    plt.close(fig)

    colors = [
        "#d62728" if row["pipeline"] in formal else "#7f7f7f"
        for _, row in df.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        df["train_time_sec"],
        df["f1_macro"],
        s=120,
        c=colors,
        alpha=0.85,
        edgecolors="black",
    )
    for _, row in df.iterrows():
        ax.annotate(
            row["pipeline"],
            (row["train_time_sec"], row["f1_macro"]),
            fontsize=8,
            xytext=(6, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("training time (s)")
    ax.set_ylabel("macro F1")
    ax.set_xscale("log")
    ax.set_title("Hold-out macro F1 vs training time (red = formal 3)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_all_pipelines_f1_vs_traintime_scatter.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        df["inference_time_per_1k_sec"],
        df["f1_macro"],
        s=120,
        c=colors,
        alpha=0.85,
        edgecolors="black",
    )
    for _, row in df.iterrows():
        ax.annotate(
            row["pipeline"],
            (row["inference_time_per_1k_sec"], row["f1_macro"]),
            fontsize=8,
            xytext=(6, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("inference time per 1000 test rows (s)")
    ax.set_ylabel("macro F1")
    ax.set_title("Hold-out macro F1 vs inference speed")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_all_pipelines_f1_vs_inference_scatter.png", dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train/evaluate Nemotron task_type classifiers; write metrics and figures."
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data") / "train_with_task_type.csv"
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--text-mode",
        choices=["prompt", "prompt_answer", "prompt_tail", "prompt_payload"],
        default="prompt_payload",
        help="Default prompt_payload = final asked item only (harder, assignment-grade spread).",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Stratified K-fold for CV statistics; set 0 to skip (faster).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory for CSV/JSON metrics and PNG plots (default: figures/).",
    )
    args = parser.parse_args()

    # 1) Data: annotated CSV from CoderGym (or download raw URL into --data).
    if args.download or not args.data.exists():
        download_if_needed(args.data)

    texts, y = load_dataset(args.data, text_mode=args.text_mode)
    X = texts.astype(str).values

    le = LabelEncoder()
    y_enc = le.fit_transform(y.values)

    # 2) Stratified hold-out; CV below uses full X/y with clones per fold.
    X_train, X_test, y_train_enc, y_test_enc = train_test_split(
        X,
        y_enc,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_enc,
    )

    pipes = build_pipelines(random_state=args.seed)
    rows: list[dict[str, Any]] = []

    # 3) Train each pipeline: optional stratified CV on full data, then hold-out evaluate_one.
    for name, estimator in pipes.items():
        print(f"\n=== Training: {name} ===")
        cv_stats: dict[str, float] = {}
        if args.cv_splits >= 2:
            cv_stats = cross_validate_stratified(
                clone(estimator), X, y_enc, args.cv_splits, args.seed
            )
            print(
                f"(CV) macro-F1: {cv_stats['cv_f1_macro_mean']:.4f} "
                f"+/- {cv_stats['cv_f1_macro_std']:.4f}"
            )

        row = evaluate_one(
            name,
            clone(estimator),
            X_train,
            y_train_enc,
            X_test,
            y_test_enc,
            le.classes_,
        )
        row.update(cv_stats)
        rows.append(row)
        print(row["classification_report"])
        print(
            f"hold-out accuracy={row['accuracy']:.4f} f1_macro={row['f1_macro']:.4f} "
            f"train={row['train_time_sec']:.3f}s infer={row['inference_time_sec']:.4f}s"
        )

    df = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 4) Export metrics/figures (CSV/JSON + PNG under output-dir).
    export_df = df.drop(
        columns=["classification_report", "y_pred"],
        errors="ignore",
    )
    # Stable, readable CSV (training loop order is arbitrary).
    if "pipeline" in export_df.columns:
        export_df = export_df.sort_values("pipeline").reset_index(drop=True)
    export_df.to_csv(args.output_dir / "metrics_summary.csv", index=False)

    manifest = {
        "text_mode": args.text_mode,
        "test_size": args.test_size,
        "random_seed": args.seed,
        "cv_splits": args.cv_splits,
        "n_rows": int(len(X)),
        "class_names": [str(c) for c in le.classes_],
    }
    with open(args.output_dir / "evaluation_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(
        args.output_dir / "classification_reports.json", "w", encoding="utf-8"
    ) as f:
        json.dump(
            {r["pipeline"]: r["classification_report"] for r in rows}, f, indent=2
        )

    plot_comparison(df, FORMAL_THREE, args.output_dir)
    plot_cv_f1_formal(df, FORMAL_THREE, args.cv_splits, args.output_dir)
    plot_formal_confusion(rows, FORMAL_THREE, y_test_enc, le.classes_, args.output_dir)

    formal_only = df[df["pipeline"].isin(FORMAL_THREE)].drop(
        columns=["classification_report", "y_pred"],
        errors="ignore",
    )
    _order_map = {n: i for i, n in enumerate(FORMAL_THREE)}
    formal_only = formal_only.copy()
    formal_only["_order"] = formal_only["pipeline"].map(_order_map)
    formal_only = formal_only.sort_values("_order").drop(columns=["_order"])
    with open(
        args.output_dir / "formal_three_metrics.json", "w", encoding="utf-8"
    ) as f:
        json.dump(formal_only.to_dict(orient="records"), f, indent=2)

    print(f"\nSaved metrics and figures under: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
