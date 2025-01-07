import numba
import numpy as np
import math

@numba.jit(nopython=True)
def factorial(n):
    """
    A simple factorial for n >= 0, used for calc_weight.
    """
    if n < 2:
        return 1.0
    result = 1.0
    for i in range(2, n + 1):
        result *= i
    return result

@numba.jit(nopython=True)
def calc_weight(U, V):
    """
    Shapley weighting function:
       calc_weight(U, V) = U! * (V - U - 1)! / V!
    with boundary checks. 
    """
    num = factorial(U) * factorial(V - U - 1)
    den = factorial(V)
    return num / den

class Tree:
    """
    A simple tree representation compatible with the 
    interventional recursion. We store:

    - children_left[node], children_right[node]:  the left/right child indices
                                                  or -1 if leaf
    - features[node], thresholds[node]:           the feature index + threshold
                                                  used at each node
    - values[node]:                               the leaf value for that node
    - max_node:                                   the total number of nodes
    - n_features:                                 total number of features (V)
    """

    def __init__(self, children_left, children_right, features, thresholds, values, n_features):
        self.children_left = children_left.astype(np.int32)
        self.children_right = children_right.astype(np.int32)
        self.features = features.astype(np.int32)
        self.thresholds = thresholds
        self.values = values
        self.n_features = n_features

    def is_leaf(self, node):
        return (self.children_left[node] == -1 and 
                self.children_right[node] == -1)

@numba.jit(nopython=True)
def interventional_recurse(
    node_index,
    U,
    V,
    xlist,
    clist,
    x,
    c,
    children_left,
    children_right,
    features,
    thresholds,
    values,
    phi
):
    """
    The two-path recursion from Algorithm 3 in https://www.nature.com/articles/s42256-019-0138-9. 

    Parameters
    ----------
    node_index : int
        Current node id
    U : int
        # of features that are 'in' so far
    V : int
        total # of features (remains constant)
    xlist : int[:]
        counters for how many times each feature is forced to follow x
    clist : int[:]
        counters for how many times each feature is forced to follow c
    x : float[:]
        The instance we want to explain
    c : float[:]
        Current reference (baseline) instance
    children_left, children_right : int[:]
        node arrays for left/right children
    features : int[:]
        feature index used at each node
    thresholds : float[:]
        threshold used at each node
    values : float[:]
        value of each node if leaf, or 0 otherwise
    phi : float[:]
        The shap attribution array, shape (V,)

    Returns
    -------
    (pos, neg) : (float, float)
        partial contributions from the recursion
    """
    # print("phi:", phi)
    # print("node_index", node_index)
    # print("U, V:", U, V)
    # If leaf node => compute possible contribution at leaf
    if children_left[node_index] == -1 and children_right[node_index] == -1:
        vj = values[node_index]
        pos = 0.0
        neg = 0.0
        if U == 0:
            return (0.0, 0.0)
        if U != 0:
            pos += calc_weight(U-1, V) * vj
        if U != V:
            neg -= calc_weight(U, V) * vj
        # print("U, V, vj, pos, neg:", U, V, vj, pos, neg)
        return (pos, neg)

    # It's an internal node => gather feature, threshold
    dj = features[node_index]
    tj = thresholds[node_index]

    a_j = children_left[node_index]
    b_j = children_right[node_index]

    xdj = x[dj]
    cdj = c[dj]

    # If xlist[dj] > 0 => forcibly follow x
    if xlist[dj] > 0:
        if xdj > tj:
            pos_ch, neg_ch = interventional_recurse(
                b_j, U, V, xlist, clist, x, c, 
                children_left, children_right,
                features, thresholds, values,
                phi
            )
        else:
            pos_ch, neg_ch = interventional_recurse(
                a_j, U, V, xlist, clist, x, c,
                children_left, children_right,
                features, thresholds, values,
                phi
            )
        return (pos_ch, neg_ch)

    # If clist[dj] > 0 => forcibly follow c
    if clist[dj] > 0:
        if cdj > tj:
            pos_ch, neg_ch = interventional_recurse(
                b_j, U, V, xlist, clist, x, c,
                children_left, children_right,
                features, thresholds, values,
                phi
            )
        else:
            pos_ch, neg_ch = interventional_recurse(
                a_j, U, V, xlist, clist, x, c,
                children_left, children_right,
                features, thresholds, values,
                phi
            )
        return (pos_ch, neg_ch)
    
    # print("xdj, cdj, tj:", xdj, cdj, tj)

    # If neither is forced => check if x & c both go left or both go right
    both_right = (xdj > tj) and (cdj > tj)
    both_left = (xdj <= tj) and (cdj <= tj)

    if both_right:
        # print("both_right")
        pos_ch, neg_ch = interventional_recurse(
            b_j, U, V, xlist, clist, x, c,
            children_left, children_right,
            features, thresholds, values,
            phi
        )
        return (pos_ch, neg_ch)

    if both_left:
        # print("both_left")
        pos_ch, neg_ch = interventional_recurse(
            a_j, U, V, xlist, clist, x, c,
            children_left, children_right,
            features, thresholds, values,
            phi
        )
        return (pos_ch, neg_ch)

    # x and c differ => "split path"
    # 1) x -> right, c -> left
    if (xdj > tj) and (cdj <= tj):
        # print("x -> right, c -> left")
        # first recursion (posx, negx) => x forced right
        xlist[dj] += 1
        (posx, negx) = interventional_recurse(
            b_j, U+1, V+1, xlist, clist, x, c,
            children_left, children_right,
            features, thresholds, values,
            phi
        )
        xlist[dj] -= 1

        # second recursion => c forced left
        clist[dj] += 1
        (posc, negc) = interventional_recurse(
            a_j, U, V+1, xlist, clist, x, c,
            children_left, children_right,
            features, thresholds, values,
            phi
        )
        clist[dj] -= 1

        phi[dj] += (posx + negc)
        return (posx + posc, negx + negc)

    # 2) x -> left, c -> right
    if (xdj <= tj) and (cdj > tj):
        # print("x -> left, c -> right")
        xlist[dj] += 1
        (posx, negx) = interventional_recurse(
            a_j, U+1, V+1, xlist, clist, x, c,
            children_left, children_right,
            features, thresholds, values,
            phi
        )
        xlist[dj] -= 1

        clist[dj] += 1
        (posc, negc) = interventional_recurse(
            b_j, U, V+1, xlist, clist, x, c,
            children_left, children_right,
            features, thresholds, values,
            phi
        )
        clist[dj] -= 1

        phi[dj] += (posx + negc)
        return (posx + posc, negx + negc)

    print("Should not reach here!")
    # fallback
    return (0.0, 0.0)


class TreeInterventionalExplainer:
    """
    Interventional Tree SHAP Explainer
    using the two-path recursion (Algorithm 3)
    for RandomForestRegressor from sklearn.
    """

    def __init__(self, model, **kwargs):
        """
        model : A trained RandomForestRegressor
        """

        # If the model is indeed a RandomForestRegressor, 
        # build our own list of Tree objects
        if str(type(model)).endswith("sklearn.ensemble._forest.RandomForestRegressor'>"):
            self.trees = []
            n_features = model.n_features_in_  # total # of features in data
            for estimator in model.estimators_:
                t_ = estimator.tree_
                tree_obj = Tree(
                    children_left=t_.children_left,
                    children_right=t_.children_right,
                    features=t_.feature,
                    thresholds=t_.threshold,
                    values=t_.value[:, 0, 0],   # shape (n_nodes, 1, 1) => flatten
                    n_features=n_features
                )
                self.trees.append(tree_obj)
        else:
            raise ValueError("This explainer currently only supports RandomForestRegressor.")

    def shap_values(self, X, refset, **kwargs):
        """
        Calculate interventional SHAP values for each row in X,
        averaged over all reference vectors in refset, 
        and averaged across all trees in the forest.

        Parameters
        ----------
        X : numpy.ndarray (N x M)
            Data for which we want shap values.
        refset : list or numpy.ndarray
            A set of baseline vectors c to average over.

        Returns
        -------
        phi : numpy.ndarray
            shape (N, M), the shap values (not including a separate bias column).
            If you'd like to add a bias term, you can do so similarly 
            to the path-based code (phi has M+1 shape). 
        """
        # convert dataframes if needed
        if str(type(X)).endswith("'pandas.core.frame.DataFrame'>") or \
           str(type(X)).endswith("'pandas.core.series.Series'>"):
            X = X.values

        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        n_samples, n_features = X.shape
        print("n_samples, n_features:", n_samples, n_features)
        print("num_trees:", len(self.trees))

        refset = np.array(refset)
        if len(refset.shape) == 1:
            refset = refset.reshape(1, -1)

        phi = np.zeros((n_samples, n_features), dtype=np.float64)

        # For each sample i in X
        for i in range(n_samples):
            # Accumulate across all trees, then average at the end
            row_phi = np.zeros(n_features, dtype=np.float64)
            for t in self.trees:
                # For each reference c in refset
                sum_phi_over_c = np.zeros(n_features, dtype=np.float64)

                for c_idx in range(refset.shape[0]):
                    xlist = np.zeros(n_features, dtype=np.int64)
                    clist = np.zeros(n_features, dtype=np.int64)

                    # RECURSE from root node = 0, U=0, V=0
                    interventional_recurse(
                        0, 
                        0, 
                        0,
                        xlist, 
                        clist,
                        X[i, :],
                        refset[c_idx, :],
                        t.children_left,
                        t.children_right,
                        t.features,
                        t.thresholds,
                        t.values,
                        sum_phi_over_c
                    )
                # Average over all c in refset
                sum_phi_over_c /= refset.shape[0]

                # Add to row_phi
                row_phi += sum_phi_over_c

            # average over all trees
            row_phi /= len(self.trees)

            phi[i, :] = row_phi

        return phi