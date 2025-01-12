import numba
import numpy as np
import math

from weight_utils import w_shapley, w_banzhaf, w_beta_shapley, w_weighted_banzhaf

def tree_prob(baseline, explicand, model, weighting="shapley"):
    explainer = TreeProbExplainer(model, weighting=weighting)
    phi = explainer.prob_values(
        X=explicand,
        refset=baseline,
    )
    return phi

# @numba.jit(nopython=True)
def calc_weight(U, V, weighting="banzhaf"):
    """
    Calculate the weight of Shapley, Banzhaf, or Beta Shapley value.

    Parameters
    ----------
    U : int
        # of features that matched the foreground sample along the path
    V : int
        total # of unique features encountered along the path
    weighting : str
        The type of weight to calculate. One of 'shapley', 'banzhaf', 'beta'.
    Returns 
    -------
    float
        The weight of the Shapley value.
    """
    if weighting == "shapley":
        return w_shapley(V, U)
    elif weighting == "banzhaf":
        return w_banzhaf(V)
    elif weighting == "beta_shapley":
        alpha = 1
        beta = 1
        return w_beta_shapley(V, U, alpha, beta)
    elif weighting == "weighted_banzhaf":
        weight = 0.5
        return w_weighted_banzhaf(V, weight)
    else:
        raise ValueError("Invalid weighting type.")

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
        self.children_left = children_left
        self.children_right = children_right
        self.features = features
        self.thresholds = thresholds
        self.values = values
        self.n_features = n_features

    def is_leaf(self, node):
        return (self.children_left[node] == -1 and 
                self.children_right[node] == -1)

# @numba.jit(nopython=True)
def prob_recurse(
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
    phi,
    weighting="shapley"
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
        # print("node_index, U, V:", node_index, U, V)
        vj = values[node_index]
        # print("values", values)
        pos = 0.0
        neg = 0.0
        if V == 0:
            return (0.0, 0.0)
        else:
            pos += calc_weight(U-1, V, weighting) * vj
            neg -= calc_weight(U, V, weighting) * vj
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
            pos_ch, neg_ch = prob_recurse(
                b_j, U, V, xlist, clist, x, c, 
                children_left, children_right,
                features, thresholds, values,
                phi
            )
        else:
            pos_ch, neg_ch = prob_recurse(
                a_j, U, V, xlist, clist, x, c,
                children_left, children_right,
                features, thresholds, values,
                phi
            )
        return (pos_ch, neg_ch)

    # If clist[dj] > 0 => forcibly follow c
    if clist[dj] > 0:
        if cdj > tj:
            pos_ch, neg_ch = prob_recurse(
                b_j, U, V, xlist, clist, x, c,
                children_left, children_right,
                features, thresholds, values,
                phi
            )
        else:
            pos_ch, neg_ch = prob_recurse(
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
        # print("a_j, b_j:", a_j, b_j)
        pos_ch, neg_ch = prob_recurse(
            b_j, U, V, xlist, clist, x, c,
            children_left, children_right,
            features, thresholds, values,
            phi
        )
        return (pos_ch, neg_ch)

    if both_left:
        # print("both_left")
        # print("a_j, b_j:", a_j, b_j)
        pos_ch, neg_ch = prob_recurse(
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
        (posx, negx) = prob_recurse(
            b_j, U+1, V+1, xlist, clist, x, c,
            children_left, children_right,
            features, thresholds, values,
            phi
        )
        xlist[dj] -= 1

        # second recursion => c forced left
        clist[dj] += 1
        (posc, negc) = prob_recurse(
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
        (posx, negx) = prob_recurse(
            a_j, U+1, V+1, xlist, clist, x, c,
            children_left, children_right,
            features, thresholds, values,
            phi
        )
        xlist[dj] -= 1

        clist[dj] += 1
        (posc, negc) = prob_recurse(
            b_j, U, V+1, xlist, clist, x, c,
            children_left, children_right,
            features, thresholds, values,
            phi
        )
        clist[dj] -= 1
        
        # print("posx, negc:", posx, negc)
        phi[dj] += (posx + negc)
        # print("phi[dj]:", phi[dj])
        return (posx + posc, negx + negc)

    print("Should not reach here!")
    # fallback
    return (0.0, 0.0)


class TreeProbExplainer:
    """
    Interventional Tree Probabilistic Value Explainer
    using the two-path recursion (Algorithm 3)
    for RandomForestRegressor from sklearn.
    """

    def __init__(self, model, weighting, **kwargs):
        """
        model : A trained RandomForestRegressor
        """
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
        self.weighting = weighting

    def prob_values(self, X, refset, **kwargs):
        """
        Calculate interventional prob values for each row in X,
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
        # print("n_samples, n_features:", n_samples, n_features)
        # print("num_trees:", len(self.trees))

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
                    prob_recurse(
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
                        sum_phi_over_c,
                        weighting=self.weighting
                    )
                # Average over all c in refset
                sum_phi_over_c /= refset.shape[0]

                # Add to row_phi
                row_phi += sum_phi_over_c

            # average over all trees
            row_phi /= len(self.trees)

            phi[i, :] = row_phi

        return phi