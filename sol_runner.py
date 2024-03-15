import PyEDCR


def run():
    epsilons = [0.1 * i for i in range(2, 3)]
    test_bool = False

    for eps in epsilons:
        print('#' * 25 + f'eps = {eps}' + '#' * 50)
        edcr = PyEDCR.EDCR(epsilon=eps,
                           main_model_name='vit_b_16',
                           combined=True,
                           loss='BCE',
                           lr=0.0001,
                           num_epochs=20,
                           include_inconsistency_constraint=False)
        edcr.print_metrics(test=test_bool, prior=True)

        edcr.run_learning_pipeline()
        edcr.run_error_detection_application_pipeline(test=test_bool)
