# Logic Tensor Network
import ltn
import torch
import data_preprocessing


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


def compute_sat_normally(logits_to_predicate,
                         prediction, labels_coarse, labels_fine):
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
    Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    And = ltn.Connective(ltn.fuzzy_ops.AndProd())
    Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
    Forall = ltn.Quantifier(
        ltn.fuzzy_ops.AggregPMeanError(p=4), quantifier="f")
    SatAgg = ltn.fuzzy_ops.SatAgg()

    fine_label_dict = {name: label for label, name in enumerate(data_preprocessing.fine_grain_classes)}
    coarse_label_dict = {name: label + len(data_preprocessing.fine_grain_classes) for label, name in
                         enumerate(data_preprocessing.coarse_grain_classes)}
    labels_fine = labels_fine.detach().to('cpu')
    labels_coarse = labels_coarse.detach().to('cpu') + len(data_preprocessing.fine_grain_classes)

    # Define constant
    l = {}
    num_labels = len(data_preprocessing.fine_grain_classes) + len(data_preprocessing.coarse_grain_classes)
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

    sat_agg_list = []
    sat_agg_label = []

    # Coarse labels: for all x[i], x[i] -> l[i]

    for i in coarse_label_dict.values():
        if x_variables[i].value.numel() != 0:
            sat_agg_label.append(
                f'for all (coarse label) x[{i}], x[{i}] -> l[{i}]')
            sat_agg_list.append(
                Forall(x_variables[i], logits_to_predicate(x_variables[i], l[i])))

    # Coarse Label: for all x[coarse], - {x[coarse] and x[different coarse]}

    for i in coarse_label_dict.values():
        for j in coarse_label_dict.values():
            if i != j:
                sat_agg_list.append(Forall(x, Not(And(logits_to_predicate(x, l[i]), logits_to_predicate(x, l[j])))))

    # Fine to coarse label: for all x[fine], x[fine] and x[correspond coarse]

    for label_fine, label_coarse in data_preprocessing.fine_to_course_idx.items():
        label_coarse = label_coarse + len(fine_label_dict)
        if x_variables[label_fine].value.numel() != 0:
            sat_agg_list.append(Forall(x_variables[label_fine],
                                       And(logits_to_predicate(x_variables[label_fine], l[label_fine]),
                                           logits_to_predicate(x_variables[label_fine], l[label_coarse])))
                                )

    # Fine labels: for all x[i], x[i] -> l[i]

    for i in fine_label_dict.values():
        if x_variables[i].value.numel() != 0:
            sat_agg_list.append(
                Forall(x_variables[i], logits_to_predicate(x_variables[i], l[i])))

    # Coarse Label: for all x[coarse], - {x[coarse] and x[different coarse]}

    for i in fine_label_dict.values():
        for j in fine_label_dict.values():
            if i != j:
                sat_agg_list.append(Forall(x, Not(And(logits_to_predicate(x, l[i]), logits_to_predicate(x, l[j])))))

    sat_agg = SatAgg(
        *sat_agg_list
    )

    return sat_agg
