# Logic Tensor Network
import ltn
import numpy as np
import torch
import data_preprocessing
import conditions
import rules
import typing


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
                    binary_pred: typing.Dict[data_preprocessing.Label, np.array],
                    conds: set[conditions.Condition],
                    device: torch.device):
    any_condition_satisfied = torch.zeros_like(cond_fine_data).detach().to('cpu')
    for cond in conds:
        any_condition_satisfied |= torch.tensor(
            cond(fine_data=cond_fine_data.detach().to('cpu').numpy(),
                 coarse_data=cond_coarse_data.detach().to('cpu').numpy(),
                 secondary_fine_data=cond_second_fine_data.detach().to(
                     'cpu').numpy() if cond_second_fine_data is not None else None,
                 secondary_coarse_data=cond_second_coarse_data.detach().to(
                     'cpu').numpy() if cond_second_coarse_data is not None else None,
                 binary_data=binary_pred if binary_pred is not None else None))
    return any_condition_satisfied.to(device)


def true_predicate(examples: torch.tensor,
                   true_data: torch.tensor,
                   device: torch.device):
    # Ensure shapes are compatible
    assert examples.shape[0] == true_data.shape[0], "Prediction and true_data must have the same batch size."

    # Get the indices of the maximum values along the second dimension (num_labels)
    pred_indices = examples.argmax(dim=1)

    # Compare the predicted indices with the true labels, resulting in a boolean tensor
    # Convert booleans to 1.0 and 0.0 using torch.float for consistency
    return torch.where(pred_indices == true_data[0], 1., 0.).to(device)


def compute_sat_normally(preprocessor: data_preprocessing.DataPreprocessor,
                         logits_to_predicate: torch.nn.Module,
                         train_pred_fine_batch: torch.tensor,
                         train_pred_coarse_batch: torch.tensor,
                         train_true_fine_batch: torch.tensor,
                         train_true_coarse_batch: torch.tensor,
                         original_train_pred_fine_batch: torch.tensor,
                         original_train_pred_coarse_batch: torch.tensor,
                         secondary_train_pred_fine_batch: torch.tensor,
                         secondary_train_pred_coarse_batch: torch.tensor,
                         binary_pred: typing.Dict[data_preprocessing.Label, np.array],
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
    g_fine = preprocessor.granularities['fine']
    g_coarse = preprocessor.granularities['coarse']

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
    for l in (list(preprocessor.fine_grain_labels.values()) +
              list(preprocessor.coarse_grain_labels.values())):
        Conds_predicate[l] = ltn.Predicate(func=lambda x, prediction: conds_predicate(
            examples=x,
            prediction=prediction,
            cond_fine_data=original_train_pred_fine_batch,
            cond_coarse_data=original_train_pred_coarse_batch,
            cond_second_fine_data=secondary_train_pred_fine_batch,
            cond_second_coarse_data=secondary_train_pred_coarse_batch,
            binary_pred=binary_pred,
            conds=error_detection_rules[l].C_l,
            device=device))

    True_predicate = ltn.Predicate(func=lambda x, train_true_batch: true_predicate(
        examples=x,
        true_data=train_true_batch,
        device=device)
                                   )

    # Define constant: already done in data_preprocessing.py
    pred_fine_data = ltn.Constant(train_pred_fine_batch)
    pred_coarse_data = ltn.Constant(train_pred_coarse_batch)
    true_fine_data = ltn.Constant(train_true_fine_batch)
    true_coarse_data = ltn.Constant(train_true_coarse_batch)
    label_one_hot = {}
    for l in preprocessor.fine_grain_labels.values():
        one_hot = torch.zeros(len(preprocessor.fine_grain_classes_str))
        one_hot[l.index] = 1.0
        label_one_hot[l] = ltn.Constant(one_hot.to(device))

    for l in preprocessor.coarse_grain_labels.values():
        one_hot = torch.zeros(len(preprocessor.coarse_grain_classes_str))
        one_hot[l.index] = 1.0
        label_one_hot[l] = ltn.Constant(one_hot.to(device))

    # Define variables
    x_variables = {}
    x_fine = ltn.Variable("x_fine", train_pred_fine_batch)
    x_coarse = ltn.Variable("x_coarse", train_pred_coarse_batch)

    for l in preprocessor.fine_grain_labels.values():
        x_variables[l] = ltn.Variable(
            str(l), train_pred_fine_batch[train_pred_fine_batch == l.index])
    for l in preprocessor.coarse_grain_labels.values():
        x_variables[l] = ltn.Variable(
            str(l), train_pred_coarse_batch[train_pred_coarse_batch == l.index])

    sat_agg_list = []

    # Detection Rule: pred_i(w) and not(true_i(w)) <- pred_i(w) and disjunction DC_i(cond_j(w))
    # error_i(w) = pred_i(w) and not(true_i(w))

    for l in preprocessor.fine_grain_labels.values():
        confidence_score = (
            Forall(x_fine,
                   Implies(
                       And(logits_to_predicate(x_fine, label_one_hot[l]),
                           Conds_predicate[l](x_fine, pred_fine_data)),
                       And(
                           Not(True_predicate(x_fine, true_fine_data)),
                           logits_to_predicate(x_fine, label_one_hot[l]))
                   )
                   ))
        sat_agg_list.append(confidence_score)

    for l in preprocessor.coarse_grain_labels.values():
        confidence_score = (
            Forall(x_coarse,
                   Implies(
                       And(logits_to_predicate(x_coarse, label_one_hot[l]),
                           Conds_predicate[l](x_coarse, pred_coarse_data)),
                       And(
                           Not(True_predicate(x_coarse, true_coarse_data)),
                           logits_to_predicate(x_coarse, label_one_hot[l]))
                   )
                   ))
        sat_agg_list.append(confidence_score)

    sat_agg = SatAgg(
        *sat_agg_list
    )

    return sat_agg

# def compute_sat_testing_value(logits_to_predicate: torch.nn.Module,
#                               pred_fine_batch: torch.tensor,
#                               pred_coarse_batch: torch.tensor,
#                               true_fine_batch: torch.tensor,
#                               true_coarse_batch: torch.tensor,
#                               original_pred_fine_batch: torch.tensor,
#                               original_pred_coarse_batch: torch.tensor,
#                               original_secondary_pred_fine_batch: torch.tensor,
#                               original_secondary_pred_coarse_batch: torch.tensor,
#                               error_detection_rules: dict[data_preprocessing.Label, rules.ErrorDetectionRule],
#                               device: torch.device):
#     """
#         compute satagg function for rules
#         argument:
#           - logits_to_predicate: get the satisfaction of a variable given the label
#           - prediction: output of fine tuner,
#           - labels_coarse, labels_fine: ground truth of coarse and fine label
#           - fine_to_coarse: dictionary mapping fine-grain class to coarse-grain class
#
#         return:
#           sat_agg: sat_agg for all the rules
#         """
#     g_fine = data_preprocessing.granularities['fine']
#     g_coarse = data_preprocessing.granularities['coarse']
#
#     # Define predicate
#     Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
#     And = ltn.Connective(ltn.fuzzy_ops.AndProd())
#     Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
#     Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
#     Forall = ltn.Quantifier(
#         ltn.fuzzy_ops.AggregPMeanError(p=4), quantifier="f")
#     SatAgg = ltn.fuzzy_ops.SatAgg()
#
#     # Cond predicate l: 1 if example satisfy any cond in DC_l and 0 otherwise]
#     Conds_predicate = {}
#     for l in (list(data_preprocessing.get_labels(g_fine).values()) +
#               list(data_preprocessing.get_labels(g_coarse).values())):
#         Conds_predicate[l] = ltn.Predicate(func=lambda x, prediction: conds_predicate(
#             examples=x,
#             prediction=prediction,
#             cond_fine_data=original_pred_fine_batch,
#             cond_coarse_data=original_pred_coarse_batch,
#             cond_second_fine_data=original_secondary_pred_fine_batch,
#             cond_second_coarse_data=original_secondary_pred_coarse_batch,
#             conds=error_detection_rules[l].C_l,
#             device=device))
#
#     True_predicate = ltn.Predicate(func=lambda x, true_batch: true_predicate(
#         examples=x,
#         true_data=true_batch,
#         device=device)
#                                    )
#
#     # Define constant: already done in data_preprocessing.py
#     pred_fine_data = ltn.Constant(pred_fine_batch)
#     pred_coarse_data = ltn.Constant(pred_coarse_batch)
#     true_fine_data = ltn.Constant(true_fine_batch)
#     true_coarse_data = ltn.Constant(true_coarse_batch)
#     label_one_hot = {}
#     for l in data_preprocessing.get_labels(g_fine).values():
#         one_hot = torch.zeros(len(data_preprocessing.fine_grain_classes_str))
#         one_hot[l.index] = 1.0
#         label_one_hot[l] = ltn.Constant(one_hot.to(device))
#
#     for l in data_preprocessing.get_labels(g_coarse).values():
#         one_hot = torch.zeros(len(data_preprocessing.coarse_grain_classes_str))
#         one_hot[l.index] = 1.0
#         label_one_hot[l] = ltn.Constant(one_hot.to(device))
#
#     # Define variables
#     x_variables = {}
#     x_fine = ltn.Variable("x_fine", pred_fine_batch)
#     x_coarse = ltn.Variable("x_coarse", pred_coarse_batch)
#
#     for l in data_preprocessing.get_labels(g=g_fine).values():
#         x_variables[l] = ltn.Variable(
#             str(l), pred_fine_batch[pred_fine_batch == l.index])
#     for l in data_preprocessing.get_labels(g=g_coarse).values():
#         x_variables[l] = ltn.Variable(
#             str(l), pred_coarse_batch[pred_coarse_batch == l.index])
#
#     sat_agg_list = []
#     sat_agg_average_score = 0
#
#     # Detection Rule: pred_i(w) and not(true_i(w)) <- pred_i(w) and disjunction DC_i(cond_j(w))
#     # error_i(w) = pred_i(w) and not(true_i(w))
#
#     for l in data_preprocessing.get_labels(g_fine).values():
#         confidence_score = (
#             Forall(x_fine,
#                    Implies(
#                        And(logits_to_predicate(x_fine, label_one_hot[l]),
#                            Conds_predicate[l](x_fine, pred_fine_data)),
#                        And(
#                            Not(True_predicate(x_fine, true_fine_data)),
#                            logits_to_predicate(x_fine, label_one_hot[l]))
#                    )
#                    ))
#         sat_agg_list.append(confidence_score)
#         sat_agg_average_score += confidence_score.value.detach().item()
#     print(f'for all w in operational data, i in fine grain classes, rule \n'
#           f'pred_i(w) and not(true_i(w)) <- pred_i(w) and disjunction DC_i(cond_j(w)) \n'
#           f'has average score {sat_agg_average_score / len(data_preprocessing.fine_grain_classes_str)}')
#
#     sat_agg_average_score = 0
#     for l in data_preprocessing.get_labels(g_coarse).values():
#         confidence_score = (
#             Forall(x_coarse,
#                    Implies(
#                        And(logits_to_predicate(x_coarse, label_one_hot[l]),
#                            Conds_predicate[l](x_coarse, pred_coarse_data)),
#                        And(
#                            Not(True_predicate(x_coarse, true_coarse_data)),
#                            logits_to_predicate(x_coarse, label_one_hot[l]))
#                    )
#                    ))
#         sat_agg_list.append(confidence_score)
#         sat_agg_average_score += confidence_score.value.detach().item()
#     print(f'for all w in operational data, i in coarse grain classes, rule \n'
#           f'pred_i(w) and not(true_i(w)) <- pred_i(w) and disjunction DC_i(cond_j(w)) \n'
#           f'has average score {sat_agg_average_score / len(data_preprocessing.coarse_grain_classes_str)}')
