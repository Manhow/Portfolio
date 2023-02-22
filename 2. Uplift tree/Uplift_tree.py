import numpy as np


class Node:
    """Base class
    """

    def __init__(self, n_items, deltaP):
        self.n_items = n_items
        self.deltaP = deltaP
        self.split_feat = None
        self.split_threshold = None
        self.left = None
        self.right = None


class UpliftTreeRegressor():
    """Uplfit Tree class with DeltaDeltaP splitting criterion 
    """

    def __init__(self,
                 max_depth: int = 3,
                 min_samples_leaf: int = 1000,
                 min_samples_leaf_treated: int = 300,
                 min_samples_leaf_control: int = 300,
                 ):
        # do something
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_leaf_treated = min_samples_leaf_treated
        self.min_samples_leaf_control = min_samples_leaf_control

    def _DeltaP(self, y: np.ndarray, treatment: np.ndarray) -> float:
        """Fucntion calculates DeltaP 

        Args:
            y (np.ndarray): target variable
            treatment (np.ndarray): treatment flag (0 or 1)

        Returns:
            float: DeltaP
        """
        P_treat = np.mean(y[treatment == 1])
        P_control = np.mean(y[treatment == 0])
        return float(P_treat - P_control)

    def _best_split(self, X: np.ndarray, y: np.ndarray, treatment: np.ndarray):
        """Function looks for the best split on each feature of the input data
        X in order to maximize the uplift between two groups (treated and control).
        The best split is determined by computing the difference in uplift between
        the left and right sides of the split (DeltaP_left and DeltaP_right) and 
        choosing the threshold that produces the maximum absolute difference. 
        The best feature and threshold are stored in best_feat and best_thr. 
        The code also applies constraints on the minimum number of samples in 
        a leaf node and in treated/control groups to ensure that the split is valid.

        Args:
            X (np.ndarray): Features (n * k)
            y (np.ndarray): Target variable (n)
            treatment (_type_): Treatment/control flag

        Returns:
            best_feat (int/None): Index of the best feature
            best_thr (float/None): Best threshold value
        """
        best_split_voc = {}
        best_thr_voc = {}
        best_feat, best_thr = None, None
        for n_feature in range(self.n_features_):
            unique_values = np.unique(X[:, n_feature])
            threshold_options = np.unique(np.percentile(X[:, n_feature], [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])) if len(
                unique_values) > 10 else np.unique(np.percentile(X[:, n_feature], [10, 50, 90]))
            max_curr_uplift = []
            thr_true_list = []
            for threshold in threshold_options:
                indices_left = X[:, n_feature] <= threshold
                y_left, treat_left = y[indices_left], treatment[indices_left]
                y_right, treat_right = y[~indices_left], treatment[~indices_left]
                if y_left.size < self.min_samples_leaf \
                        or y_right.size < self.min_samples_leaf \
                        or y_left[treat_left == 1].size < self.min_samples_leaf_treated \
                        or y_left[treat_left == 0].size < self.min_samples_leaf_control \
                        or y_right[treat_right == 1].size < self.min_samples_leaf_treated \
                        or y_right[treat_right == 0].size < self.min_samples_leaf_control:
                    continue
                thr_true_list.append(threshold)
                DeltaP_left = self._DeltaP(
                    y_left, treat_left)
                DeltaP_right = self._DeltaP(
                    y_right, treat_right)
                curr_uplift = np.abs(DeltaP_left - DeltaP_right)
                max_curr_uplift.append(curr_uplift)
            if max_curr_uplift:
                max_idx = np.argmax(max_curr_uplift)
                best_split_voc[n_feature] = max_curr_uplift[max_idx]
                best_thr_voc[n_feature] = thr_true_list[max_idx]
        try:
            best_feat, _ = max(
                best_split_voc.items(), key=lambda x: x[1])
            best_thr = best_thr_voc[best_feat]
        except:
            pass
        return best_feat, best_thr

    def _build_tree(self, X: np.ndarray, y: np.ndarray, treatment: np.ndarray, depth=0):
        """Using recursion, the function builds new nodes by splitting
        the data (left and right) with _best_split function until the maximum depth is reached.

        Args:
            X (np.ndarray): Features (n * k)
            y (np.ndarray): Target variable (n)
            treatment (np.ndarray): Treatment/control flag
            depth (int, optional): Tree dept (hyperparameter). Defaults to 0.

        Returns:
            object: Current node with computed fields.
        """
        node = Node(
            n_items=y.size,
            deltaP=self._DeltaP(y, treatment),
        )
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y, treatment)
            if idx is not None:
                indices_left = X[:, idx] <= thr
                X_left, y_left, treat_left = X[indices_left], y[indices_left], treatment[indices_left]
                X_right, y_right, treat_right = X[~indices_left], y[~indices_left], treatment[~indices_left]

                node.split_feat = idx
                node.split_threshold = thr
                node.left = self._build_tree(
                    X_left, y_left, treat_left, depth + 1)
                node.right = self._build_tree(
                    X_right, y_right, treat_right, depth + 1)
        return node

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            treatment: np.ndarray
            ) -> None:
        """Fit method.

        Args:
            X (np.ndarray): Features (n * k)
            y (np.ndarray): Target variable (n)
            treatment (np.ndarray): Treatment/control flag
        """
        self.n_features_ = X.shape[1]
        self.tree_ = self._build_tree(X, y, treatment)
        pass

    def predict(self, X: np.ndarray):
        """Predict method that applies the _predict method
        on each element in the input array X and returns a list of predictions.

        Args:
            X (np.ndarray): X_test

        Returns:
            list[Iterable]: Predicted uplift
        """
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs):
        """This code is a part of a decision tree prediction,
        where it traverses the tree based on input features, 
        and returns the prediction of the reached leaf node.

        Args:
            inputs (np.ndarray): X_test

        Returns:
            float: DeltaP value in the current Node
        """
        node = self.tree_
        while node.left:
            if inputs[node.split_feat] < node.split_threshold:
                node = node.left
            else:
                node = node.right
        return node.deltaP

    def display_node(self, node, depth):
        """Visualization method. The code then checks if the node has a left
        and/or right child node. If it does, it will recursively call the
        display_node function on that child node, increasing the depth by 1. 
        This will result in a tree-like structure being printed out
        to the console, with each node's information indented according to 
        its depth in the tree.

        Args:
            node (_type_): Node object
            depth (int): Tree depth
        """
        indent = "    "
        print(indent*depth, "depth: ", depth)
        print(indent*depth, "n_items: ", node.n_items)
        print(indent*depth, "deltaP: ", node.deltaP)
        print(indent*depth, "split_feat: ", node.split_feat)
        print(indent*depth, "split_threshold: ", node.split_threshold)
        print(indent*depth, '-----------------')
        if node.left:
            print(indent*depth, "Left child:")
            self.display_node(node.left, depth+1)
        if node.right:
            print(indent*depth, "Right child:")
            self.display_node(node.right, depth+1)
