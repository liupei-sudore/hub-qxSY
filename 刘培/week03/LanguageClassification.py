#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12 类语种分类模型训练脚本（scikit-learn，CPU 秒级可训练）

默认数据：HuggingFace `papluca/language-identification`（20 语种，每种 3500 条训练样本）
默认 12 类：zh ja ar ru en fr de es pt it nl pl

用法：
  python train_langid.py                          # 默认训练 12 类
  python train_langid.py --limit-per-lang 1000    # 快速试跑
  python train_langid.py --data my.csv            # 用自有数据（含 text,label 两列）
  python train_langid.py --predict "Hello there" "你好世界" --predict-only
"""

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

# ----------------------------- 配置 -----------------------------
DEFAULT_LANGS = "zh,ja,ar,ru,en,fr,de,es,pt,it,nl,pl"   # 12 类
MAX_CHARS = 512                                          # 单条文本截断长度

LANG_NAMES = {
    "zh": "中文", "ja": "日语", "ko": "韩语", "ar": "阿拉伯语", "ru": "俄语",
    "en": "英语", "fr": "法语", "de": "德语", "es": "西班牙语", "pt": "葡萄牙语",
    "it": "意大利语", "nl": "荷兰语", "pl": "波兰语", "tr": "土耳其语",
    "vi": "越南语", "th": "泰语", "hi": "印地语", "el": "希腊语",
    "sw": "斯瓦希里语", "ur": "乌尔都语", "bg": "保加利亚语",
}

NAME2CODE = {c: c for c in LANG_NAMES}
NAME2CODE.update({v: k for k, v in LANG_NAMES.items()})
NAME2CODE.update({
    "chinese": "zh", "mandarin": "zh", "zh-cn": "zh", "zh-tw": "zh", "汉语": "zh",
    "english": "en", "eng": "en",
    "japanese": "ja", "日本語": "ja", "jp": "ja",
    "korean": "ko", "한국어": "ko", "kr": "ko",
    "arabic": "ar", "العربية": "ar",
    "russian": "ru", "русский": "ru",
    "french": "fr", "français": "fr", "fra": "fr",
    "german": "de", "deutsch": "de", "ger": "de",
    "spanish": "es", "español": "es", "spa": "es",
    "portuguese": "pt", "português": "pt",
    "italian": "it", "italiano": "it",
    "dutch": "nl", "nederlands": "nl",
    "polish": "pl", "polski": "pl",
})

_WS_RE = re.compile(r"\s+")


# ----------------------------- 文本预处理 -----------------------------
def clean_text(s) -> str:
    """NFC 归一化 + 小写 + 空白折叠 + 截断。train / predict 必须一致。"""
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = unicodedata.normalize("NFC", s).lower()
    s = _WS_RE.sub(" ", s).strip()
    return s[:MAX_CHARS]


def normalize_label(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    return NAME2CODE.get(s.lower(), NAME2CODE.get(s, s.lower()))


# ----------------------------- 数据读取 -----------------------------
def read_any(path, text_col=None, label_col=None) -> pd.DataFrame:
    p = Path(path)
    if p.is_dir():  # 目录结构：data/zh/xxx.txt, data/en/yyy.txt
        rows = [{"text": f.read_text(encoding="utf-8", errors="ignore"),
                 "label": f.parent.name}
                for f in sorted(p.rglob("*")) if f.is_file() and f.suffix.lower() in {".txt", ".text"}]
        if not rows:
            raise ValueError(f"目录 {p} 下未找到 txt 文件")
        return pd.DataFrame(rows)

    sep = "\t" if p.suffix.lower() in {".tsv", ".tab"} else ","
    df = pd.read_csv(p, sep=sep, on_bad_lines="skip")
    tc = text_col or ("text" if "text" in df.columns else df.columns[0])
    lc = label_col or next((c for c in ("label", "labels", "lang") if c in df.columns), df.columns[1])
    return df[[tc, lc]].rename(columns={tc: "text", lc: "label"})


def read_hf(dataset_name, split) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("缺少依赖：pip install datasets；或改用 --data 指定本地数据")
    print(f"  从 HuggingFace 加载 {dataset_name} [{split}] ...")
    try:
        ds = load_dataset(dataset_name, split=split)
    except Exception as e:
        sys.exit(f"加载失败：{e}\n请检查网络，或使用 --data 指定本地数据文件")
    cols = ds.column_names
    tc = "text" if "text" in cols else cols[0]
    lc = next((c for c in ("labels", "label", "lang") if c in cols), cols[1])
    return pd.DataFrame({"text": ds[tc], "label": ds[lc]})


def prepare(args, langs):
    df = read_any(args.data, args.text_col, args.label_col) if args.data \
        else read_hf(args.hf_dataset, args.hf_split)

    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].map(normalize_label)
    df = df[df["label"].isin(langs)]
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() >= 2]

    if args.limit_per_lang:
        df = pd.concat([g.sample(n=min(len(g), args.limit_per_lang), random_state=args.seed)
                        for _, g in df.groupby("label")]).reset_index(drop=True)

    counts = df["label"].value_counts()
    missing = [l for l in langs if l not in counts.index]
    if missing:
        print(f"  ⚠️  数据中缺少以下语种，已跳过：{missing}")
    langs = [l for l in langs if l in counts.index]
    df = df[df["label"].isin(langs)].reset_index(drop=True)

    print(f"  样本总数 {len(df)}，语种 {len(langs)} 类：")
    for l in langs:
        print(f"    {l:>3}  {LANG_NAMES.get(l, ''):<6} {counts[l]:>6}")
    return df, langs


# ----------------------------- 模型 -----------------------------
def build_pipeline(args) -> Pipeline:
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=args.min_df,
        max_features=args.max_features, sublinear_tf=True,
        lowercase=False, dtype=np.float32,
    )
    blocks = [("char", char_vec)]
    if args.use_word:
        blocks.append(("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=args.min_df,
            max_features=100000, sublinear_tf=True, lowercase=False, dtype=np.float32)))

    feats = FeatureUnion(blocks) if len(blocks) > 1 else blocks[0][1]
    clf = (LinearSVC(C=args.C) if args.clf == "svm"
           else LogisticRegression(C=args.C, max_iter=2000, solver="lbfgs"))

    return Pipeline([("feat", feats), ("norm", Normalizer()), ("clf", clf)])


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def evaluate(model, X, y, labels, title):
    pred = model.predict(X)
    acc = accuracy_score(y, pred)
    macro_f1 = f1_score(y, pred, average="macro")
    report = classification_report(y, pred, labels=labels, digits=4, zero_division=0)
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=labels), index=labels, columns=labels)

    print(f"\n--- {title} ---")
    print(f"accuracy = {acc:.4f}   macro-F1 = {macro_f1:.4f}   n = {len(y)}")
    print(report)
    print("混淆矩阵（行=真实，列=预测）:")
    print(cm.to_string())
    return {"accuracy": acc, "macro_f1": macro_f1, "report": report, "cm": cm}


def run_predict(model_path, texts):
    bundle = joblib.load(model_path)
    model, labels = bundle["model"], bundle["labels"]
    names = bundle.get("lang_names", {})

    X = [clean_text(t) for t in texts]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
    else:
        proba = softmax(model.decision_function(X))

    print("\n--- 预测结果 ---")
    for t, p in zip(texts, proba):
        i = int(np.argmax(p))
        code = labels[i]
        print(f"  {t[:40]!r:<45} -> {code:<3} ({names.get(code, code)})  conf={p[i]:.4f}")


# ----------------------------- 主流程 -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="12 类语种分类模型训练")
    p.add_argument("--data", default=None, help="本地数据：csv/tsv（含 text,label）或按语种分目录的 txt 目录")
    p.add_argument("--text-col", default=None, help="文本列名（默认自动识别）")
    p.add_argument("--label-col", default=None, help="标签列名（默认自动识别）")
    p.add_argument("--hf-dataset", default="papluca/language-identification")
    p.add_argument("--hf-split", default="train")
    p.add_argument("--langs", default=DEFAULT_LANGS, help="逗号分隔的语种代码")
    p.add_argument("--limit-per-lang", type=int, default=None, help="每类最多采样条数（快速试验）")
    p.add_argument("--clf", choices=["lr", "svm"], default="lr")
    p.add_argument("--C", type=float, default=5.0)
    p.add_argument("--max-features", type=int, default=300000)
    p.add_argument("--min-df", type=int, default=2)
    p.add_argument("--use-word", action="store_true", help="额外融合词级 1-2gram 特征")
    p.add_argument("--out", default="models/langid")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--predict", nargs="+", default=None, help="训练后用这些文本做预测")
    p.add_argument("--predict-only", action="store_true", help="跳过训练，直接用已有模型预测")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    langs = [normalize_label(x) for x in args.langs.split(",") if x.strip()]
    out_dir = Path(args.out)
    model_path = out_dir / "langid.joblib"

    if args.predict_only:
        if not model_path.exists():
            sys.exit(f"模型不存在：{model_path}")
        if not args.predict:
            sys.exit("--predict-only 需要配合 --predict \"文本\" 使用")
        run_predict(model_path, args.predict)
        return

    t0 = time.time()
    print("[1/4] 加载数据 ...")
    df, labels = prepare(args, langs)
    if len(labels) < 2:
        sys.exit("有效语种不足 2 类，请检查数据或 --langs 参数")

    print("\n[2/4] 划分数据集（train/val/test = 8:1:1，分层抽样）...")
    train_df, tmp = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=args.seed)
    val_df, test_df = train_test_split(tmp, test_size=0.5, stratify=tmp["label"], random_state=args.seed)
    print(f"  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    print(f"\n[3/4] 训练模型（clf={args.clf}, C={args.C}, word_feat={args.use_word}）...")
    model = build_pipeline(args)
    model.fit(train_df["text"].to_numpy(), train_df["label"].to_numpy())
    print(f"  训练耗时 {time.time() - t0:.1f}s")

    print("\n[4/4] 评估 ...")
    res_val = evaluate(model, val_df["text"].to_numpy(), val_df["label"].to_numpy(), labels, "验证集")
    res_test = evaluate(model, test_df["text"].to_numpy(), test_df["label"].to_numpy(), labels, "测试集")

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": labels, "max_chars": MAX_CHARS,
                 "lang_names": {l: LANG_NAMES.get(l, l) for l in labels},
                 "version": 1}, model_path, compress=3)
    (out_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.txt").write_text(
        f"accuracy(test)={res_test['accuracy']:.4f}\nmacro_f1(test)={res_test['macro_f1']:.4f}\n\n"
        + res_test["report"] + "\n\n" + res_test["cm"].to_string(), encoding="utf-8")
    res_test["cm"].to_csv(out_dir / "confusion_matrix.csv", encoding="utf-8-sig")

    print(f"\n✅ 模型已保存：{model_path}")
    print(f"   评估报告：{out_dir / 'report.txt'}")
    print(f"   混淆矩阵：{out_dir / 'confusion_matrix.csv'}")

    if args.predict:
        run_predict(model_path, args.predict)


if __name__ == "__main__":
    main()
