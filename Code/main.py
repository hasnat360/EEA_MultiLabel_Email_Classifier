import random
import numpy as np

from preprocess import get_input_data, de_duplication, noise_remover, create_chained_cols, translate_to_en
from embeddings import get_tfidf_embd
from modelling import chained_model_predict, hierarchical_model_predict
from Config import Config

random.seed(Config.SEED)
np.random.seed(Config.SEED)


def _banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


if __name__ == "__main__":
    _banner("PHASE 1 — Loading and Preprocessing Data")
    df = get_input_data()
    print(f"  Loaded {len(df)} records from {len(Config.DATA_FILES)} file(s): {', '.join(Config.DATA_FILES)}")
    df = de_duplication(df)
    df = noise_remover(df)
    print(f"  After preprocessing: {len(df)} records remain.")

    print("\n  Data overview:")
    for col in [Config.Y1, Config.Y2, Config.Y3, Config.Y4]:
        n_valid = df[col].notna().sum()
        n_classes = df[col].dropna().nunique()
        print(f"    {col}: {n_valid} valid / {len(df)} total | {n_classes} unique classes")

    _banner("PHASE 2 — Translation (optional)")
    df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    print("  Translation step complete.")
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].astype("U")
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].astype("U")

    _banner("PHASE 3 — Creating Chained Label Columns")
    df = create_chained_cols(df)
    for chain_name, cols in Config.CHAINED_TARGETS.items():
        col_out = f"y_{chain_name}"
        n_valid = df[col_out].notna().sum()
        n_classes = df[col_out].dropna().nunique()
        print(f"  {col_out} ({Config.CHAIN_SEPARATOR.join(cols)}): {n_valid} valid rows | {n_classes} unique combined labels")

    _banner("PHASE 4 — Computing TF-IDF Embeddings")
    X = get_tfidf_embd(df)
    print(f"  Embedding matrix shape: {X.shape}")

    dc1_results = chained_model_predict(X, df)
    dc2_results, dc2_model_count = hierarchical_model_predict(X, df)

    print("\n" + "*" * 70)
    print("*  FINAL COMPARISON: Design Choice 1 vs Design Choice 2           *")
    print("*" * 70)

    print("\n  DESIGN CHOICE 1 — Chained Multi-Output:")
    print(f"  Total models: {len(dc1_results)}")
    for chain_name, info in dc1_results.items():
        print(f"  {chain_name}: {info['label']:<32} Accuracy = {info['accuracy']:.4f} ({info['n_classes']} classes)")

    print("\n  DESIGN CHOICE 2 — Hierarchical Modelling:")
    print(f"  Total models trained: {dc2_model_count}")
    for r in dc2_results:
        if r["accuracy"] is not None:
            print(f"  L{r['level']} | {r['parent']:<45} -> {r['target']}: Accuracy = {r['accuracy']:.4f}")
        else:
            print(f"  L{r['level']} | {r['parent']:<45} -> {r['target']}: {r.get('note', 'N/A')}")

    print("\n  KEY ARCHITECTURAL DIFFERENCES:")
    print(f"  DC1 uses {len(dc1_results)} model(s); DC2 uses {dc2_model_count} model(s).")
    print("  DC1 evaluates combined labels as a single multi-class target.")
    print("  DC2 routes test samples using previous predictions before the next classifier runs.")
    print("  DC2 therefore reflects error propagation across the hierarchy more faithfully.")

    print("\n" + "=" * 70)
    print("  Pipeline complete.")
    print("=" * 70 + "\n")
