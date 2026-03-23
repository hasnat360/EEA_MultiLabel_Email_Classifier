import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from model.randomforest import RandomForest
from data_model import Data, FilteredData
from Config import Config
from utils import remove_low_frequency_classes


def _train_and_evaluate(data, model_name: str) -> RandomForest:
    rf = RandomForest(model_name=model_name, embeddings=data.embeddings, y=data.y)
    rf.train(data)
    rf.predict(data.X_test)
    rf.print_results(data)
    return rf


def chained_model_predict(X: np.ndarray, df: pd.DataFrame) -> dict:
    print("\n" + "=" * 70)
    print("  DESIGN CHOICE 1 — CHAINED MULTI-OUTPUT CLASSIFICATION")
    print("=" * 70)
    print("  One RandomForest per chain level. Combined label = one multi-class target.\n")

    results = {}
    for chain_name, cols in Config.CHAINED_TARGETS.items():
        target_col = f"y_{chain_name}"
        label = Config.CHAIN_SEPARATOR.join(cols)
        print(f"  {'#' * 66}")
        print(f"  Chain Level : {chain_name}  |  Target: {label}")
        print(f"  {'#' * 66}")
        data = Data(X, df, target_col)
        if not data.is_valid():
            print(f"  [SKIP] Insufficient data for '{chain_name}'.\n")
            continue

        print(f"  Training samples : {len(data.y_train)}")
        print(f"  Test samples     : {len(data.y_test)}")
        print(f"  Unique classes   : {len(set(data.y_train))}\n")

        model = _train_and_evaluate(data, f"RF_{chain_name}")
        acc = accuracy_score(data.y_test, model.predictions)
        results[chain_name] = {
            "label": label,
            "accuracy": acc,
            "train_size": len(data.y_train),
            "test_size": len(data.y_test),
            "n_classes": len(set(data.y_train)),
        }
        print()

    print("\n" + "=" * 70)
    print("  DESIGN CHOICE 1 — SUMMARY")
    print("=" * 70)
    print(f"  {'Chain':<12} {'Target':<30} {'Accuracy':>10} {'Classes':>10}")
    print(f"  {'-'*12} {'-'*30} {'-'*10} {'-'*10}")
    for cn, info in results.items():
        print(f"  {cn:<12} {info['label']:<30} {info['accuracy']:>10.4f} {info['n_classes']:>10}")
    print()
    return results


def _prepare_level1_split(X: np.ndarray, df: pd.DataFrame, target_col: str):
    mask = df[target_col].notna() & (df[target_col].astype(str).str.strip() != "")
    X_f = X[mask.values]
    df_f = df.loc[mask].reset_index(drop=True)
    X_f, df_f = remove_low_frequency_classes(df_f, X_f, target_col, Config.MIN_CLASS_COUNT)
    y = df_f[target_col].values
    idx = np.arange(len(df_f))
    tr_idx, te_idx = train_test_split(
        idx,
        test_size=Config.TEST_SIZE,
        random_state=Config.SEED,
        stratify=y,
    )
    return X_f, df_f, tr_idx, te_idx


def _fit_branch_model(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, model_name: str):
    data = FilteredData(X_train, X_test, y_train, y_test)
    model = _train_and_evaluate(data, model_name)
    return model, accuracy_score(y_test, model.predictions)


def hierarchical_model_predict(X: np.ndarray, df: pd.DataFrame):
    levels = Config.HIERARCHICAL_LEVELS
    all_results = []
    model_count = 0

    print("\n" + "=" * 70)
    print("  DESIGN CHOICE 2 — HIERARCHICAL MODELLING")
    print("=" * 70)
    print("  Previous model predictions are used to route test samples to the next level.")
    print(f"  Hierarchy: {' -> '.join(levels)}\n")

    l1_col, l2_col, l3_col = levels
    X_l1, df_l1, tr1, te1 = _prepare_level1_split(X, df, l1_col)
    train_df_l1 = df_l1.iloc[tr1].reset_index(drop=True)
    test_df_l1 = df_l1.iloc[te1].reset_index(drop=True)
    X_train_l1 = X_l1[tr1]
    X_test_l1 = X_l1[te1]
    y_train_l1 = train_df_l1[l1_col].values
    y_test_l1 = test_df_l1[l1_col].values

    print(f"  {'#' * 66}")
    print(f"  LEVEL 1 — Classifying {l1_col} on the full dataset")
    print(f"  {'#' * 66}")
    print(f"  Training: {len(y_train_l1)} | Test: {len(y_test_l1)}")
    print(f"  Classes : {sorted(set(y_train_l1))}\n")

    data_l1 = FilteredData(X_train_l1, X_test_l1, y_train_l1, y_test_l1)
    rf_l1 = _train_and_evaluate(data_l1, f"RF_L1_{l1_col}")
    model_count += 1
    pred_l1 = rf_l1.predictions
    acc_l1 = accuracy_score(y_test_l1, pred_l1)
    all_results.append({
        "level": 1,
        "parent": "ALL",
        "target": l1_col,
        "accuracy": acc_l1,
        "train": len(y_train_l1),
        "test": len(y_test_l1),
        "classes": sorted(set(y_train_l1)),
        "routing": "full dataset",
    })

    print(f"\n  {'#' * 66}")
    print(f"  LEVEL 2 — Classifying {l2_col} with routing from predicted {l1_col}")
    print(f"  {'#' * 66}")

    train_classes_l1 = sorted(set(y_train_l1))
    l2_predictions = pd.Series(index=test_df_l1.index, dtype=object)

    for parent_cls in train_classes_l1:
        print(f"\n  --- Predicted {l1_col} = '{parent_cls}' → classifying {l2_col} ---")
        train_mask = train_df_l1[l1_col] == parent_cls
        branch_train_df = train_df_l1.loc[train_mask].reset_index(drop=True)
        X_branch_train = X_train_l1[train_mask.values]

        valid_train = branch_train_df[l2_col].notna() & (branch_train_df[l2_col].astype(str).str.strip() != "")
        branch_train_df = branch_train_df.loc[valid_train].reset_index(drop=True)
        X_branch_train = X_branch_train[valid_train.values]

        X_branch_train, branch_train_df = remove_low_frequency_classes(
            branch_train_df,
            X_branch_train,
            l2_col,
            Config.MIN_BRANCH_CLASS_COUNT,
        )

        y_branch_train = branch_train_df[l2_col].values
        if len(branch_train_df) < Config.MIN_SUBSET_SIZE:
            note = f"skipped — too few training samples ({len(branch_train_df)})"
            print(f"  [SKIP] {note}")
            all_results.append({"level": 2, "parent": f"pred_{l1_col}={parent_cls}", "target": l2_col, "accuracy": None, "train": len(branch_train_df), "test": 0, "classes": [], "note": note})
            continue
        if len(set(y_branch_train)) < 2:
            note = f"skipped — only {len(set(y_branch_train))} training class(es)"
            print(f"  [SKIP] {note}")
            all_results.append({"level": 2, "parent": f"pred_{l1_col}={parent_cls}", "target": l2_col, "accuracy": None, "train": len(branch_train_df), "test": 0, "classes": sorted(set(y_branch_train)), "note": note})
            continue

        test_mask = pd.Series(pred_l1 == parent_cls, index=test_df_l1.index)
        routed_idx = test_df_l1.index[test_mask.values]
        branch_test_df = test_df_l1.loc[routed_idx].copy()
        X_branch_test = X_test_l1[test_mask.values]
        valid_test = branch_test_df[l2_col].notna() & (branch_test_df[l2_col].astype(str).str.strip() != "")
        valid_routed_idx = branch_test_df.index[valid_test.values]
        branch_test_df = branch_test_df.loc[valid_routed_idx].reset_index(drop=True)
        X_branch_test = X_branch_test[valid_test.values]
        y_branch_test = branch_test_df[l2_col].values

        if len(branch_test_df) == 0:
            note = "skipped — no routed test samples"
            print(f"  [SKIP] {note}")
            all_results.append({"level": 2, "parent": f"pred_{l1_col}={parent_cls}", "target": l2_col, "accuracy": None, "train": len(branch_train_df), "test": 0, "classes": sorted(set(y_branch_train)), "note": note})
            continue

        print(f"  Training: {len(y_branch_train)} | Routed test: {len(y_branch_test)}")
        print(f"  Classes : {sorted(set(y_branch_train))}\n")
        rf_l2, acc_l2 = _fit_branch_model(X_branch_train, y_branch_train, X_branch_test, y_branch_test, f"RF_L2_{l1_col}={parent_cls}_{l2_col}")
        model_count += 1
        all_results.append({
            "level": 2,
            "parent": f"pred_{l1_col}={parent_cls}",
            "target": l2_col,
            "accuracy": acc_l2,
            "train": len(y_branch_train),
            "test": len(y_branch_test),
            "classes": sorted(set(y_branch_train)),
            "routing": f"predicted {l1_col}",
        })
        l2_predictions.loc[valid_routed_idx] = rf_l2.predictions

    print(f"\n  {'#' * 66}")
    print(f"  LEVEL 3 — Classifying {l3_col} with routing from predicted {l1_col} and predicted {l2_col}")
    print(f"  {'#' * 66}")

    for parent_cls in train_classes_l1:
        train_mask_l1 = train_df_l1[l1_col] == parent_cls
        branch_train_df_l1 = train_df_l1.loc[train_mask_l1].reset_index(drop=True)
        X_branch_train_l1 = X_train_l1[train_mask_l1.values]
        valid_train_l2 = branch_train_df_l1[l2_col].notna() & (branch_train_df_l1[l2_col].astype(str).str.strip() != "")
        branch_train_df_l1 = branch_train_df_l1.loc[valid_train_l2].reset_index(drop=True)
        X_branch_train_l1 = X_branch_train_l1[valid_train_l2.values]
        X_branch_train_l1, branch_train_df_l1 = remove_low_frequency_classes(
            branch_train_df_l1,
            X_branch_train_l1,
            l2_col,
            Config.MIN_BRANCH_CLASS_COUNT,
        )
        if len(branch_train_df_l1) == 0:
            continue

        for child_cls in sorted(branch_train_df_l1[l2_col].dropna().unique()):
            print(f"\n    --- Predicted {l1_col}='{parent_cls}', predicted {l2_col}='{child_cls}' → classifying {l3_col} ---")
            train_mask_l2 = branch_train_df_l1[l2_col] == child_cls
            branch_train_df_l2 = branch_train_df_l1.loc[train_mask_l2].reset_index(drop=True)
            X_branch_train_l2 = X_branch_train_l1[train_mask_l2.values]

            valid_train_l3 = branch_train_df_l2[l3_col].notna() & (branch_train_df_l2[l3_col].astype(str).str.strip() != "")
            branch_train_df_l2 = branch_train_df_l2.loc[valid_train_l3].reset_index(drop=True)
            X_branch_train_l2 = X_branch_train_l2[valid_train_l3.values]
            X_branch_train_l2, branch_train_df_l2 = remove_low_frequency_classes(
                branch_train_df_l2,
                X_branch_train_l2,
                l3_col,
                Config.MIN_BRANCH_CLASS_COUNT,
            )
            y_branch_train_l3 = branch_train_df_l2[l3_col].values

            if len(branch_train_df_l2) < 4:
                note = f"skipped — too few training samples ({len(branch_train_df_l2)})"
                print(f"    [SKIP] {note}")
                all_results.append({"level": 3, "parent": f"pred_{l1_col}={parent_cls}, pred_{l2_col}={child_cls}", "target": l3_col, "accuracy": None, "train": len(branch_train_df_l2), "test": 0, "classes": [], "note": note})
                continue
            if len(set(y_branch_train_l3)) < 2:
                note = f"skipped — only {len(set(y_branch_train_l3))} training class(es)"
                print(f"    [SKIP] {note}")
                all_results.append({"level": 3, "parent": f"pred_{l1_col}={parent_cls}, pred_{l2_col}={child_cls}", "target": l3_col, "accuracy": None, "train": len(branch_train_df_l2), "test": 0, "classes": sorted(set(y_branch_train_l3)), "note": note})
                continue

            routed_mask = (pd.Series(pred_l1, index=test_df_l1.index) == parent_cls) & (l2_predictions == child_cls)
            branch_test_df_l2 = test_df_l1.loc[routed_mask.fillna(False)].reset_index(drop=True)
            X_branch_test_l2 = X_test_l1[routed_mask.fillna(False).values]
            valid_test_l3 = branch_test_df_l2[l3_col].notna() & (branch_test_df_l2[l3_col].astype(str).str.strip() != "")
            branch_test_df_l2 = branch_test_df_l2.loc[valid_test_l3].reset_index(drop=True)
            X_branch_test_l2 = X_branch_test_l2[valid_test_l3.values]
            y_branch_test_l3 = branch_test_df_l2[l3_col].values

            if len(branch_test_df_l2) == 0:
                note = "skipped — no routed test samples"
                print(f"    [SKIP] {note}")
                all_results.append({"level": 3, "parent": f"pred_{l1_col}={parent_cls}, pred_{l2_col}={child_cls}", "target": l3_col, "accuracy": None, "train": len(branch_train_df_l2), "test": 0, "classes": sorted(set(y_branch_train_l3)), "note": note})
                continue

            print(f"    Training: {len(y_branch_train_l3)} | Routed test: {len(y_branch_test_l3)}")
            print(f"    Classes : {sorted(set(y_branch_train_l3))}\n")
            rf_l3, acc_l3 = _fit_branch_model(
                X_branch_train_l2,
                y_branch_train_l3,
                X_branch_test_l2,
                y_branch_test_l3,
                f"RF_L3_{l1_col}={parent_cls}_{l2_col}={child_cls}_{l3_col}",
            )
            model_count += 1
            all_results.append({
                "level": 3,
                "parent": f"pred_{l1_col}={parent_cls}, pred_{l2_col}={child_cls}",
                "target": l3_col,
                "accuracy": acc_l3,
                "train": len(y_branch_train_l3),
                "test": len(y_branch_test_l3),
                "classes": sorted(set(y_branch_train_l3)),
                "routing": f"predicted {l1_col} + predicted {l2_col}",
            })

    return all_results, model_count
