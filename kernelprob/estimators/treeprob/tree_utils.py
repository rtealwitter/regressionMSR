import xgboost
import numpy as np
import math

from .tree import Tree

def is_sklearn_random_forest(model):
    return (
        "sklearn.ensemble._forest.RandomForestRegressor" in str(type(model))
        or "sklearn.ensemble._forest.RandomForestClassifier" in str(type(model))
    )

def is_xgboost_model(model):
    return xgboost is not None and (
        isinstance(model, xgboost.XGBRegressor) or isinstance(model, xgboost.XGBClassifier)
    )

def parse_xgb_tree(
        df,
        tree_id,
        booster,
        n_features=None,
        n_trees=None,
    ):
        """
        Parse a single XGBoost tree (tree_id) from the dataframe `df`
        returned by booster.trees_to_dataframe().

        Parameters
        ----------
        df : pd.DataFrame
            The entire DataFrame from model.get_booster().trees_to_dataframe().
        tree_id : int
            The index of the tree to parse.
        booster : xgboost.Booster
            The booster object from the trained XGBoost model. We'll use
            booster.feature_names to map from feature string -> int index.
        n_features : int, optional
            The total number of features. If None, we'll try to infer from the data.
        
        Returns
        -------
        children_left : np.ndarray of shape (num_nodes,)
        children_right : np.ndarray of shape (num_nodes,)
        features : np.ndarray of shape (num_nodes,)
        thresholds : np.ndarray of shape (num_nodes,)
        values : np.ndarray of shape (num_nodes,)
        n_features_inferred : int
        """
        if not booster.feature_names:
            raise ValueError("booster.feature_names is empty. "
                            "Either your model wasn't trained with named columns, "
                            "or you need to set booster.feature_names manually.")

        df_tree = df[df["Tree"] == tree_id].copy()
        if df_tree.empty:
            raise ValueError(f"No nodes found for tree_id={tree_id}.")

        unique_ids = df_tree["ID"].unique()
        node_mapping = {id_str: idx for idx, id_str in enumerate(sorted(unique_ids))}
        node_count = len(unique_ids)

        children_left  = np.full(node_count, -1, dtype=int)
        children_right = np.full(node_count, -1, dtype=int)
        features       = np.full(node_count, -1, dtype=int)
        thresholds     = np.full(node_count, np.nan, dtype=float)
        values         = np.zeros(node_count, dtype=float)

        maybe_max_fid = 0

        # use the booster.feature_names to map from str -> int index
        booster_feat_names = booster.feature_names

        for row in df_tree.itertuples(index=False):
            node_id_str = getattr(row, "ID")  # e.g. "0-0", "0-1"
            nid = node_mapping[node_id_str]

            feat = getattr(row, "Feature")
            if feat == "Leaf":
                # It's a leaf node
                leaf_val = getattr(row, "Leaf", float("nan"))
                if (leaf_val is None) or (isinstance(leaf_val, float) and math.isnan(leaf_val)):
                    leaf_val = getattr(row, "Gain", 0.0)
                values[nid] = float(leaf_val) * n_trees # scale by # of trees to match sklearn

                children_left[nid]  = -1
                children_right[nid] = -1

            else:
                # It's an internal node
                # Map feature name to index
                try:
                    fid = booster_feat_names.index(feat)
                except ValueError:
                    # If the feature name not found in booster_feat_names, fallback -1
                    fid = -1
                    
                features[nid] = fid
                maybe_max_fid = max(maybe_max_fid, fid)

                # threshold
                split_val = getattr(row, "Split", float("nan"))
                thresholds[nid] = float(split_val) if split_val is not None else float("nan")

                # children
                yes_str = getattr(row, "Yes", None)
                no_str  = getattr(row, "No", None)

                if isinstance(yes_str, str) and yes_str in node_mapping:
                    left_id = node_mapping[yes_str]
                else:
                    left_id = -1

                if isinstance(no_str, str) and no_str in node_mapping:
                    right_id = node_mapping[no_str]
                else:
                    right_id = -1

                children_left[nid]  = left_id
                children_right[nid] = right_id

        if n_features is None or n_features < (maybe_max_fid + 1):
            n_features_inferred = max(n_features or 0, maybe_max_fid + 1)
        else:
            n_features_inferred = n_features

        return Tree(
            children_left,
            children_right,
            features,
            thresholds,
            values,
            n_features_inferred
        )