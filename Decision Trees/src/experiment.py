from .decision_tree import DecisionTree
from .prior_probability import PriorProbability
from .metrics import precision_and_recall, confusion_matrix, f1_measure, accuracy
from .data import load_data, train_test_split

def run(data_path, learner_type, fraction):

    features, targets, attribute_names = load_data(data_path)
    
    if learner_type == 'prior_probability':
        a = PriorProbability()
        train_features, train_targets, test_features, test_targets = train_test_split(features,targets,fraction)
        a.fit(train_features,train_targets)
        results = a.predict(test_features)
        cm = confusion_matrix(test_targets,results)
        acc = accuracy(test_targets,results)
        prec, rec = precision_and_recall(test_targets,results)
        f1 = f1_measure(test_targets,results)
    elif learner_type == 'decision_tree':
        a = DecisionTree(attribute_names)
        train_features, train_targets, test_features, test_targets = train_test_split(features,targets,fraction)
        a.fit(train_features,train_targets)
        results = a.predict(test_features)
        cm = confusion_matrix(test_targets,results)
        acc = accuracy(test_targets,results)
        prec, rec = precision_and_recall(test_targets,results)
        f1 = f1_measure(test_targets,results)
    else:
        raise ValueError("Invalid Learner Type!")
    # Order of these returns must be maintained
    return cm,acc,prec,rec,f1
