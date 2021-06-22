import numpy as np

class Node():
    def __init__(self, value=None, attribute_name="root", attribute_index=None, branches=None):
        self.branches = [] if branches is None else branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.value = value

class DecisionTree():
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
        self.tree = None

    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!"
            )

    def fit(self, features, targets):
        self._check_input(features)
        attributes = [i for i in range(len(self.attribute_names))]
        self.tree = self.ID3(features,attributes,targets,None)



    def predict(self, features):
        self._check_input(features)
        prediction = []
        for row in features:
            pred = self.tree
            while pred.attribute_name != "root":
                if row[pred.attribute_index] == 1:
                    pred = pred.branches[1]
                else:
                    pred = pred.branches[0]
            prediction.append(pred.value)
        prediction = np.array(prediction)
        return prediction  

    def _visualize_helper(self, tree, level):
        """
        Helper function for visualize a decision tree at a given level of recursion.
        """
        tab_level = "  " * level
        val = tree.value if tree.value is not None else 0
        print("%d: %s%s == %f" % (level, tab_level, tree.attribute_name, val))

    def visualize(self, branch=None, level=0):
        if not branch:
            branch = self.tree
        self._visualize_helper(branch, level)

        for branch in branch.branches:
            self.visualize(branch, level+1)

    def ID3(self,features,attributes,targets,default=None):
        r = Node()
        if all_ones(targets):
            r.value = 1
            return r
        elif all_zeros(targets):
            r.value = 0
            return r
        elif len(attributes) == 0:
            r.value = mode_value(targets)
            return r
        elif features.shape[0] == 0:
            r.value = default
            return r
        else:
            best = best_ig(features,targets,attributes)
            best_name = self.attribute_names[best]
            new_att = attributes.copy()
            new_att.remove(best)
            subset = []
            subset1 = []
            target_att = []
            target_att1 = []
            for j in range(len(targets)):
                if float(features[j,best]) == 0.0:
                    subset.append(features[j])
                    target_att.append(targets[j])
                else:
                    subset1.append(features[j])
                    target_att1.append(targets[j])
            subset,subset1,target_att,target_att1 = np.array(subset),np.array(subset1),np.array(target_att),np.array(target_att1)
            return Node(default,best_name,best,[self.ID3(subset,new_att,target_att,0),self.ID3(subset1,new_att,target_att1,1)])
                
        
                
def all_ones(array):
    for i in range(len(array)):
        if float(array[i]) == 1:
            pass
        else:
            return False
    return True

def all_zeros(array):
    for i in range(len(array)):
        if float(array[i]) == 0:
            pass
        else:
            return False
    return True

def mode_value(targets):
    pos = 0
    neg = 0
    for i in range(len(targets)):
        if float(targets[i]) == 1:
            pos += 1
        else:
            neg += 1
    if pos > neg:
        return 1
    else:
        return 0
    
def best_ig(features,targets,attributes):
    ig = {}
    for i in attributes:
        ig[i] = information_gain(features,i,targets)
    best = max(ig,key=ig.get)
    return best
    
def information_gain(features, attribute_index, targets):
    def entropy(features, targets):
        true_count = 0
        false_count = 0
        for i in range(len(targets)):
            if float(targets[i]) == 1:
                true_count += 1
            else: 
                false_count += 1
        if true_count == 0 or false_count == 0:
            return 0
        else:
            t_prob = true_count/(true_count+false_count)
            f_prob = false_count/(true_count+false_count)
            return -(t_prob*np.log2(t_prob))-(f_prob*np.log2(f_prob))
    parents_entropy = entropy(features, targets)
    features1 = []
    targets1 = []
    features2 = []
    targets2 = []
    for i in range(features.shape[0]):
        if float(features[i,int(attribute_index)]) == 1:
            features1.append(features[i,:])
            targets1.append(targets[i])
        else:
            features2.append(features[i,:])
            targets2.append(targets[i])
    features1 = np.array(features1)
    targets1 = np.array(targets1)
    features2 = np.array(features2)
    targets2 = np.array(targets2)

    child_entropy1 = entropy(features1,targets1)
    child_entropy2 = entropy(features2,targets2)
    information_gain = parents_entropy - (child_entropy1*(features1.shape[0]/features.shape[0])+(child_entropy2*(features2.shape[0]/features.shape[0])))
    return information_gain


if __name__ == '__main__':
    # construct a fake tree
    attribute_names = ['larry', 'curly', 'moe']
    decision_tree = DecisionTree(attribute_names=attribute_names)
    while len(attribute_names) > 0:
        attribute_name = attribute_names[0]
        if not decision_tree.tree:
            decision_tree.tree = Node(
                attribute_name=attribute_name,
                attribute_index=decision_tree.attribute_names.index(attribute_name),
                value=0,
                branches=[]
            )
        else:
            decision_tree.tree.branches.append(
                Node(
                    attribute_name=attribute_name,
                    attribute_index=decision_tree.attribute_names.index(attribute_name),
                    value=0,
                    branches=[]
                )
            )
        attribute_names.remove(attribute_name)
    decision_tree.visualize()
