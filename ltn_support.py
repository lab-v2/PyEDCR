# Logic Tensor Network
import ltn
import torch
import data_preprocessing
import conditions
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


def pred_to_index(examples: torch.tensor,
                  prediction: torch.tensor):
    if examples.shape[0] != prediction.shape[0]:
        raise ValueError("Input tensors must have the same number of elements in the first dimension.")

        # Create a boolean mask for exact matches using broadcasting:
    mask = (prediction == examples[:, None]).all(dim=0)  # True where element-wise comparison is True across axis 0

    # Return the indices of True elements in the mask using torch.where:
    return torch.where(mask)[0]


def conds_predicate(examples: torch.tensor,
                    prediction: torch.tensor,
                    cond_fine_data: torch.tensor,
                    cond_coarse_data: torch.tensor,
                    cond_second_fine_data: torch.tensor,
                    cond_second_coarse_data: torch.tensor,
                    conds: set[conditions.Condition]):
    any_condition_satisfied = torch.zeros_like(cond_fine_data)
    for cond in conds:
        any_condition_satisfied |= cond(fine_data=cond_fine_data,
                                        coarse_data=cond_coarse_data,
                                        secondary_fine_data=cond_second_fine_data,
                                        secondary_coarse_data=cond_second_coarse_data)
    return any_condition_satisfied


def true_predicate(examples: torch.tensor,
                   true_data: torch.tensor):
    # Ensure shapes are compatible
    assert examples.shape[0] == true_data.shape[0], "Prediction and true_data must have the same batch size."

    # Get the indices of the maximum values along the second dimension (num_labels)
    pred_indices = examples.argmax(dim=1)

    # Compare the predicted indices with the true labels, resulting in a boolean tensor
    # Convert booleans to 1.0 and 0.0 using torch.float for consistency
    return torch.where(pred_indices == true_data[0], 1., 0.)


def compute_sat_normally(logits_to_predicate: torch.nn.Module,
                         train_pred_fine_batch: torch.tensor,
                         train_pred_coarse_batch: torch.tensor,
                         train_true_fine_batch: torch.tensor,
                         train_true_coarse_batch: torch.tensor,
                         original_train_pred_fine_batch: torch.tensor,
                         original_train_pred_coarse_batch: torch.tensor,
                         original_secondary_train_pred_fine_batch: torch.tensor,
                         original_secondary_train_pred_coarse_batch: torch.tensor,
                         error_detection_rules: dict[data_preprocessing.Label, rules.ErrorDetectionRule],
                         device: torch.device):
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

    # Define predicate
    Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    And = ltn.Connective(ltn.fuzzy_ops.AndProd())
    Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
    Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
    Forall = ltn.Quantifier(
        ltn.fuzzy_ops.AggregPMeanError(p=4), quantifier="f")
    SatAgg = ltn.fuzzy_ops.SatAgg()


    # Cond predicate l: 1 if example satisfy any cond in DC_l and 0 otherwise]
    Conds_predicate = {}
    for l in (list(data_preprocessing.get_labels(g_fine).values()) +
              list(data_preprocessing.get_labels(g_coarse).values())):
        Conds_predicate[l] = ltn.Predicate(func=lambda x, prediction: conds_predicate(
            examples=x,
            prediction=prediction,
            cond_fine_data=original_train_pred_fine_batch,
            cond_coarse_data=original_train_pred_coarse_batch,
            cond_second_fine_data=original_secondary_train_pred_fine_batch,
            cond_second_coarse_data=original_secondary_train_pred_coarse_batch,
            conds=error_detection_rules[l].C_l))

    True_predicate = ltn.Predicate(func=lambda x, train_true_batch: true_predicate(
        examples=x,
        true_data=train_true_batch)
    )

    # Define constant: already done in data_preprocessing.py
    pred_fine_data = ltn.Constant(train_pred_fine_batch)
    pred_coarse_data = ltn.Constant(train_pred_coarse_batch)
    true_fine_data = ltn.Constant(train_true_fine_batch)
    true_coarse_data = ltn.Constant(train_true_coarse_batch)

    # Define variables
    x_variables = {}
    x_fine = ltn.Variable("x_fine", train_pred_fine_batch)
    x_coarse = ltn.Variable("x_coarse", train_pred_coarse_batch)

    for l in data_preprocessing.get_labels(g=g_fine).values():
        x_variables[l] = ltn.Variable(
            str(l), train_pred_fine_batch[train_pred_fine_batch == l.index])
    for l in data_preprocessing.get_labels(g=g_coarse).values():
        x_variables[l] = ltn.Variable(
            str(l), train_pred_coarse_batch[train_pred_coarse_batch == l.index])

    sat_agg_list = []
    sat_agg_label = []

    # Detection Rule: pred_i(w) and not(true_i(w)) <- pred_i(w) and disjunction DC_i(cond_j(w))
    # error_i(w) = pred_i(w) and not(true_i(w))

    for l in data_preprocessing.get_labels(g_fine).values():
        sat_agg_list.append(
            Forall(x_fine,
                   Implies(
                       And(logits_to_predicate(x_fine, l.ltn_constant),
                           Conds_predicate[l](x_fine, pred_fine_data)),
                       And(
                           Not(True_predicate(x_fine, true_fine_data)),
                           logits_to_predicate(x_fine, l.ltn_constant))
                   )
                   ))

    for l in data_preprocessing.get_labels(g_coarse).values():
        sat_agg_list.append(
            Forall(x_coarse,
                   Implies(
                       And(logits_to_predicate(x_coarse, l.ltn_constant),
                           Conds_predicate[l](x_coarse, pred_coarse_data)),
                       And(
                           Not(True_predicate(x_coarse, true_coarse_data)),
                           logits_to_predicate(x_coarse, l.ltn_constant))
                   )
                   ))

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
