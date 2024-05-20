
# Jihye Lee
from collections import Counter
import pandas as pd
import numpy as np
import seaborn as sns
#https://towardsdatascience.com/implementing-a-decision-tree-from-scratch-f5358ff9c4bb

from sklearn import datasets

from sklearn.model_selection import train_test_split

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        """
        Initialize a node in the decision tree.

        Parameters:
        - feature: Index of the feature to split on.
        - threshold: Threshold value for the split.
        - left: Left child node.
        - right: Right child node.
        - value: Value assigned to the node if it's a leaf node.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        """
        Check if the node is a leaf node.
        """
        return self.value is not None 
        
        
    

class DecisionTreeModel:

    def __init__(self, max_depth=100, criterion = 'gini', min_samples_split=2, impurity_stopping_threshold = 1):
        """
        Initialize the Decision Tree Model.

        Parameters:
        - max_depth: Maximum depth allowed for the decision tree.
        - criterion: The impurity criterion used for splitting, either 'gini' or 'entropy'.
        - min_samples_split: Minimum number of samples required to split an internal node.
        - impurity_stopping_threshold: Threshold for stopping tree growth based on impurity.
        """
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.impurity_stopping_threshold = impurity_stopping_threshold
        self.root = None

    def fit(self, X, y):
        """
        Fit the decision tree model on the training data.
        """
        self.root = self._build_tree(X, y)
        print("Done fitting")

    def predict(self, X):
        """
        Make predictions using the trained decision tree.
        """
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)    
        
    def _is_finished(self, depth):
        """
        Check if the tree-building process should be terminated based on stopping criteria.

        Parameters:
        - depth: Current depth of the tree.

        Returns:
        - Boolean indicating whether tree-building should be terminated.
        """
        # TODO: add another stopping criteria for checking if it is homogenous enough already
        # modify the signature of the method if needed
        if (depth >= self.max_depth
            or self.n_class_labels == 1
            or self.n_samples < self.min_samples_split):
            return True
        # end TODO
        return False
    
    def _is_homogenous_enough(self, y):
        """
        Check if the impurity of the target variable is low enough.

        Parameters:
        - y: Target variable values.

        Returns:
        - Boolean indicating whether the impurity is low enough.
        """
        # check if y is homogenous enough by comparing the gini or entropy is small enough
        # TODO
        result = False
        # end TODO
        return result
                              
    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.

        Parameters:
        - X: Feature matrix.
        - y: Target variable.
        - depth: Current depth of the tree.

        Returns:
        - Root node of the decision tree.
        """
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))
    
        # Stopping criteria
        if (self._is_finished(depth) or self._is_homogenous_enough(y)):
            u, counts = np.unique(y, return_counts=True)
            most_common_Label = u[np.argmax(counts)]
            return Node(value=most_common_Label)
    
        # Get best split
        rnd_feats = np.random.choice(self.n_features, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, rnd_feats)
    
        # Grow children recursively
        left_idx, right_idx = self._create_split(X[:, best_feat], best_thresh)
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feat, best_thresh, left_child, right_child)


    

    def _gini(self, y):
        """
        Calculate the Gini impurity of a target variable.

        Parameters:
        - y: Target variable values.

        Returns:
        - Gini impurity.
        """
        #TODO: 
        _, count = np.unique(y, return_counts = True)
        proportions = count / len(y)
        gini = 1 - np.sum([p**2 for p in proportions if p > 0])
        #end TODO
        return gini
    
    def _entropy(self, y):
        """
        Calculate the entropy of a target variable.

        Parameters:
        - y: Target variable values.

        Returns:
        - Entropy.
        """
        # TODO: the following won't work if y is not integer
        # make it work for the cases where y is a categorical variable
        _, count = np.unique(y, return_counts = True)
        proportions = count / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        # end TODO
        return entropy
    
    def _create_split(self, X, thresh):
        """
        Create a split in the data based on a threshold.

        Parameters:
        - X: Feature values.
        - thresh: Threshold value for the split.

        Returns:
        - Indices of samples in the left and right splits.
        """
        left_idx = np.argwhere(X <= thresh).flatten()
        right_idx = np.argwhere(X > thresh).flatten()
        return left_idx, right_idx

    def _information_gain(self, X, y, thresh):
        """
        Calculate the information gain by splitting the data at a threshold.

        Parameters:
        - X: Feature values.
        - y: Target variable values.
        - thresh: Threshold value for the split.

        Returns:
        - Information gain.
        """
        # TODO: fix the code so it can switch between the two criterion: gini and entropy 
        if self.criterion == "enropy":
            parent_loss = self._entropy(y)
            left_idx, right_idx = self._create_split(X, thresh)
            n, n_left, n_right = len(y), len(left_idx), len(right_idx)

            if n_left == 0 or n_right == 0:
                return 0

            child_loss = (
                n_left / n) * self._entropy(y[left_idx]) + (n_right / n) * self._entropy(y[right_idx])
        # else if the criterion is gini
        elif self.criterion == "gini":
            parent_loss = self._gini(y)
            left_idx, right_idx = self._create_split(X, thresh)
            n, n_left, n_right = len(y), len(left_idx), len(right_idx)

            if n_left == 0 or n_right == 0:
                return 0

            child_loss = (
                n_left / n) * self._gini(y[left_idx]) + (n_right / n) * self._gini(y[right_idx])
        # end TODO
        return parent_loss - child_loss
       
    def _best_split(self, X, y, features):
        """
        Find the best feature and threshold for splitting the data.

        Parameters:
        - X: Feature matrix.
        - y: Target variable.
        - features: Indices of features to consider for splitting.

        Returns:
        - Best feature index and threshold.
        """
        split = {'score':- 1, 'feat': None, 'thresh': None}

        for feat in features:
            X_feat = X[:, feat]
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                score = self._information_gain(X_feat, y, thresh)

                if score > split['score']:
                    split['score'] = score
                    split['feat'] = feat
                    split['thresh'] = thresh

        return split['feat'], split['thresh']
    
    def _traverse_tree(self, x, node):
        """
        Traverse the decision tree to make predictions.

        Parameters:
        - x: Input data point.
        - node: Current node in the decision tree.

        Returns:
        - Predicted value.
        """
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForestModel(object):

    def __init__(self, n_estimators, criterion='gini', min_samples_split=2, max_depth=10):
        """
        Initialize a Random Forest model.

        Parameters:
        - n_estimators: Number of decision trees in the forest.
        - criterion: The impurity criterion to use for tree building (default is 'gini').
        - min_samples_split: The minimum number of samples required to split an internal node (default is 2).
        - max_depth: The maximum depth of the tree (default is 10).
        """
        # TODO: do NOT simply call the RandomForest model from sklearn
        self.criterion=criterion
        self.max_depth=max_depth
        self.min_samples_split = min_samples_split
        self.n_estimators = n_estimators
        self.trees = []
        # end TODO

    def fit(self, X, y):
        """
        Fit the Random Forest model to the training data.

        Parameters:
        - X: Feature matrix of the training data.
        - y: Target variable of the training data.
        """
        # TODO: call the underlying n_estimators trees fit method

        counter = 0
        while counter < self.n_estimators:
            
            clf = DecisionTreeModel(max_depth=10)
            # Create random samples with replacement for bagging
            n_samples = X.shape[0]
            random_samples = np.random.choice(n_samples, n_samples, replace=True)
            
            tempx, tempy = X[random_samples], y[random_samples]
            
            # train current decision tree
            clf.fit(tempx, tempy)
            # save to array trees 
            self.trees.append(clf)
            counter += 1
        # end TODO


    def predict(self, X):
        """
        Make predictions using the Random Forest model.

        Parameters:
        - X: Feature matrix of the data to predict.

        Returns:
        - Array of predicted target variable values.
        """
        # TODO:
        # call the predict method of the underlying trees, return the majority prediction
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(X))

        predictions = np.swapaxes(predictions, 0, 1)
        y_preds = []

        for pred in predictions:
            c = Counter(pred)
            y_preds.append(c.most_common(1)[0][0])
        return y_preds
    

def accuracy_score(y_true, y_pred):
    """
    Calculate the accuracy score.

    Parameters:
    - y_true: True target values.
    - y_pred: Predicted target values.

    Returns:
    - Accuracy score.
    """
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def confusion_matrix(y_test, y_pred):
    """
    Calculate the confusion matrix.

    Parameters:
    - y_test: True target values.
    - y_pred: Predicted target values.

    Returns:
    - Confusion matrix as a 2x2 numpy array.
    """
    # TODO: return the number of TP, TN, FP, FN
    
    a, test_count = np.unique(y_test, return_counts=True)
    b, pred_count = np.unique(y_pred, return_counts = True)
    y_copy = []
    for i in range(len(y_test)):
        val = np.where(a == y_test[i])
        y_copy.append(val[0][0])
    y_test = np.asarray(y_copy)
    pred_copy = []
    for j in range(len(y_pred)):
        val = np.where(b == y_pred[j])
        pred_copy.append(val[0][0])
    y_pred = np.asarray(pred_copy)

    false_positive = 0
    false_negative = 0

    true_positive = 0
    true_negative = 0

    for y_test, y_pred in zip(y_test, y_pred):

        if y_pred == y_test:
            if y_pred == 1:
                true_positive += 1
            else:
                true_negative += 1
        else:
            if y_pred == 1:
                false_positive += 1
            else:
                false_negative += 1
    confusion_matrix = [
        [true_positive, false_positive],
        [false_negative, true_negative],
    ]
    confusion_matrix = np.array(confusion_matrix)

    # end TODO
    return (confusion_matrix)
    

def classification_report(y_test, y_pred):
    """
    Generate a classification report.

    Parameters:
    - y_test: True target values.
    - y_pred: Predicted target values.

    Returns:
    - Dictionary containing Precision, Recall, and F1-Score.
    """
    # calculate precision, recall, f1-score and print them out
    cm = confusion_matrix(y_test, y_pred)
    TP = cm[0][0]
    FN = cm[1][0]
    FP = cm[0][1]
    # TODO:
    result = {}

    # Precision: tp / (tp + fp)
    prec = TP / (TP + FP)
    result["Precision"] = prec
    # Recall: TP / (TP + FN)
    recall = TP / (TP + FN)
    result["Recall"] = recall
    # F1-Score: 2((precision * Recall) / (Precision + Recall))
    f1 = (2 * prec * recall) / (prec + recall)
    result["F1-Score"] = f1

    # end TODO
    return(result)


def _test():

    df = pd.read_csv('breast_cancer.csv')
    
    # Call the model with integer target variable
    X = df.drop(['diagnosis'], axis=1)
    y = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    clf = DecisionTreeModel(max_depth=10, criterion='gini')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Gini Accracy: " + str(acc))

    print(classification_report(y_test,y_pred))

    clf = DecisionTreeModel(max_depth=10, criterion='entropy')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Entropy Accracy: " + str(acc))

    print(classification_report(y_test,y_pred))

    # call the model with categorical target variable
    X = df.drop(['diagnosis'], axis=1)
    y = df['diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    clf = DecisionTreeModel(max_depth=10, criterion='gini')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Gini Accracy: " + str(acc))

    print(classification_report(y_test,y_pred))

    # Run the RF model
    rfc = RandomForestModel(n_estimators=3)
    rfc.fit(X_train, y_train)
    rfc_pred = rfc.predict(X_test)
    print("RF Model")
    print(classification_report(y_test, rfc_pred))
    print(accuracy_score(y_test, rfc_pred))

if __name__ == "__main__":
    _test()
