# Logic Tensor Network
import ltn
import torch
import data_preprocessing
import conditions
import numpy as np
import rules


class LogitsToPredicate(torch.nn.Module):
    """
    This model has inside a logits model, that is a model which compute logits for the classes given an input example x.
    The idea of this model is to keep logits and probabilities separated. The logits model returns the logits for an example,
    while this model returns the probabilities given the logits model.

    In particular, it takes as input an example x and a class label d. It applies the logits model to x to get the logits.
    Then, it applies a softmax function to get the probabilities per classes. Finally, it returns only the probability related
    to the given class d.
    """

    def __init__(self):
        super(LogitsToPredicate, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, d):
        probs = self.sigmoid(x)
        out = torch.sum(probs * d, dim=1)
        return out



def compute_sat_normally(logits_to_predicate: torch.nn.Module,
                         train_pred_fine_batch: np.array,
                         train_pred_coarse_batch: np.array,
                         train_true_fine_batch: np.array,
                         train_true_coarse_batch: np.array,
                         original_train_pred_fine_batch: np.array,
                         original_train_pred_coarse_batch: np.array,
                         original_secondary_train_pred_fine_batch: np.array,
                         original_secondary_train_pred_coarse_batch: np.array,
                         error_detection_rules: dict[data_preprocessing.Label, rules.ErrorDetectionRule]):
    """
    compute satagg function for rules
    argument:
      - logits_to_predicate: get the satisfaction of a variable given the label
      - prediction: output of fine tuner, 
      - labels_coarse, labels_fine: ground truth of coarse and fine label
      - fine_to_coarse: dictionary mapping fine-grain class to coarse-grain class

    return:
      sat_agg: sat_agg for all the rules

    """
    g_fine = data_preprocessing.granularities['fine']
    g_coarse = data_preprocessing.granularities['coarse']
    Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    And = ltn.Connective(ltn.fuzzy_ops.AndProd())
    Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
    Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
    Forall = ltn.Quantifier(
        ltn.fuzzy_ops.AggregPMeanError(p=4), quantifier="f")
    SatAgg = ltn.fuzzy_ops.SatAgg()

    train_true_fine_batch = train_true_fine_batch.detach().to('cpu')
    train_true_coarse_batch = train_true_coarse_batch.detach().to('cpu')

    # Define constant: already done in data_preprocessing.py
    cond_fine = ltn.Variable("cond_fine", original_train_pred_fine_batch)
    cond_coarse = ltn.Variable("cond_coarse", original_train_pred_coarse_batch)
    cond_second_fine = ltn.Variable("cond_second_fine", original_secondary_train_pred_fine_batch)
    cond_second_coarse = ltn.Variable("cond_second_coarse", original_secondary_train_pred_coarse_batch)

    # Define variables
    x_variables = {}
    x_fine = ltn.Variable("x_fine", train_pred_fine_batch)
    x_coarse = ltn.Variable("x_coarse", train_pred_coarse_batch)

    for l in data_preprocessing.get_labels(g=g_fine).values():
        x_variables[l] = ltn.Variable(
            str(l), train_true_fine_batch[train_true_fine_batch == l.index])
    for l in data_preprocessing.get_labels(g=g_coarse).values():
        x_variables[l] = ltn.Variable(
            str(l), train_true_coarse_batch[train_true_coarse_batch == l.index])

    sat_agg_list = []
    sat_agg_label = []

    # Detection Rule: pred_i(w) and not(true_i(w)) <- pred_i(w) and disjunction DC_i(cond_j(w))
    # error_i(w) = pred_i(w) and not(true_i(w))

    for l in data_preprocessing.get_labels(g_fine).values():
        sat_agg_list.append(Forall(x_fine,
                                   Implies(
                                       And(logits_to_predicate(x_fine, l.ltn_constant),
                                                  Or(logits_to_predicate(x_fine, l_prime.ltn_constant)
                                                     for l_prime in error_detection_rules[l].)),
                                       And(
                                           Not(logits_to_predicate(x, train_true_fine_batch[])),
                                           logits_to_predicate(x_fine, l[i]))
                                   )
                                   ))

    # Correction Rule: corr_i(w) <- disj_CC_i(cond_q(w) and pred_r(w))
    for i in label_dict.values():
        sat_agg_list.append(Forall(x, Implies(Or(logits_to_predicate(x, l[q]), logits_to_predicate(x, l[r]))
                                              for q, r in CC_i), logits_to_predicate(x, ground_truth[i])))

    sat_agg = SatAgg(
        *sat_agg_list
    )

    return sat_agg


def compute_sat_testing_value(logits_to_predicate,
                              prediction, labels_coarse, labels_fine):
    """
    compute satagg function for rules
    argument:
      - logits_to_predicate: get the satisfaction of a variable given the label
      - prediction: output of fine tuner, 
      - labels_coarse, labels_fine: ground truth of coarse and fine label
      - fine_to_coarse: dictionary mapping fine-grain class to coarse-grain class

    return:
      sat_agg_label: list contain rules and its confidence value

    """
    Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    And = ltn.Connective(ltn.fuzzy_ops.AndProd())
    Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
    Forall = ltn.Quantifier(
        ltn.fuzzy_ops.AggregPMeanError(p=4), quantifier="f")
    SatAgg = ltn.fuzzy_ops.SatAgg()

    fine_label_dict = {name: label for label, name in enumerate(data_preprocessing.fine_grain_classes_str)}
    coarse_label_dict = {name: label + len(data_preprocessing.fine_grain_classes_str) for label, name in
                         enumerate(data_preprocessing.coarse_grain_classes_str)}
    labels_fine = labels_fine.detach().to('cpu')
    labels_coarse = labels_coarse.detach().to('cpu') + len(data_preprocessing.fine_grain_classes_str)

    # Define constant
    l = {}
    num_labels = len(data_preprocessing.fine_grain_classes_str) + len(data_preprocessing.coarse_grain_classes_str)
    for label in range(num_labels):
        one_hot = torch.zeros(num_labels)
        one_hot[label] = 1.0
        l[label] = ltn.Constant(one_hot, trainable=True)

    # Define variables
    x_variables = {}
    x = ltn.Variable("x", prediction)

    for name, label in fine_label_dict.items():
        x_variables[label] = ltn.Variable(
            name, prediction[labels_fine == label])
    for name, label in coarse_label_dict.items():
        x_variables[label] = ltn.Variable(
            name, prediction[labels_coarse == label])

    sat_agg_label = []

    # Rewrite the inconsistency code (Forall(x, Implies(P(x,coarse_label), Not(P(x,coarse_to_not_fine))))
    for coarse_label, i in coarse_label_dict.items():
        for fine_label, j in fine_label_dict.items():
            corresponding_coarse_label = data_preprocessing.fine_to_course_idx[j] + len(fine_label_dict)
            if (corresponding_coarse_label != i):
                satisfaction = Forall(x,
                                      Implies(logits_to_predicate(x, l[i]),
                                              Not(logits_to_predicate(x, l[j]))
                                              )
                                      )
                sat_agg_label.append([0,
                                      f"for all x, P(x, l[{coarse_label}]) imply -P(x, l[{fine_label}])",
                                      satisfaction.value.detach().item()])

    # Coarse labels: for all x[i], x[i] -> l[i]

    for coarse_label, i in coarse_label_dict.items():
        satisfaction = Forall(x_variables[i], logits_to_predicate(x_variables[i], l[i]))
        sat_agg_label.append([1,
                              f'for all {coarse_label}, P(x[{coarse_label}], l[{coarse_label}])',
                              satisfaction.value.detach().item()])

    # Coarse Label: for all x, - (P(x, l[coarse_1] and x[different coarse]}

    for coarse_label_1, i in coarse_label_dict.items():
        for coarse_label_2, j in coarse_label_dict.items():
            if i != j:
                satisfaction = Forall(x, Not(And(logits_to_predicate(x, l[i]), logits_to_predicate(x, l[j]))))
                sat_agg_label.append([2,
                                      f"for all x, - (P(x, {coarse_label_1}) and P(x,{coarse_label_2}))",
                                      satisfaction.value.detach().item()])

    # Fine labels: for all x[i], x[i] -> l[i]

    for fine_label, i in fine_label_dict.items():
        satisfaction = Forall(x_variables[i], logits_to_predicate(x_variables[i], l[i]))
        sat_agg_label.append([1,
                              f'for all {fine_label}, P(x[{fine_label}], l[{fine_label}])',
                              satisfaction.value.detach().item()])
    # Fine Label: for all x[fine], - {x[fine] and x[different fine]}

    for fine_label_1, i in fine_label_dict.items():
        for fine_label_2, j in fine_label_dict.items():
            if i != j:
                satisfaction = Forall(x, Not(And(logits_to_predicate(x, l[i]), logits_to_predicate(x, l[j]))))
                sat_agg_label.append([2,
                                      f"for all x, - (P(x, {fine_label_1}) and P(x,{fine_label_2}))",
                                      satisfaction.value.detach().item()])

    return sat_agg_label
