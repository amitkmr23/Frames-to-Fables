import numpy as np

class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._build_tree(np.array(X), np.array(y), depth=0)

    def predict(self, X):
        return [self._predict_one(row, self.tree) for row in np.array(X)]

    def _gini(self, y):
        classes = np.unique(y)
        impurity = 1
        for c in classes:
            p = np.sum(y == c) / len(y)
            impurity -= p ** 2
        return impurity

    def _best_split(self, X, y):
        best_gain = 0
        best_feat, best_val = None, None
        current_impurity = self._gini(y)

        n_features = X.shape[1]

        for feature in range(n_features):
            values = np.unique(X[:, feature])
            for val in values:
                left_idx = X[:, feature] <= val
                right_idx = X[:, feature] > val
                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue
                p = len(y[left_idx]) / len(y)
                gain = current_impurity - p * self._gini(y[left_idx]) - (1 - p) * self._gini(y[right_idx])
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feature
                    best_val = val
        return best_feat, best_val

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.bincount(y).argmax()

        feat, val = self._best_split(X, y)
        if feat is None:
            return np.bincount(y).argmax()

        left_idx = X[:, feat] <= val
        right_idx = X[:, feat] > val

        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return (feat, val, left, right)

    def _predict_one(self, row, node):
        if not isinstance(node, tuple):
            return node
        feat, val, left, right = node
        if row[feat] <= val:
            return self._predict_one(row, left)
        else:
            return self._predict_one(row, right)
