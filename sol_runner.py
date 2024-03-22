import PyEDCR
import data_preprocessing
import vit_pipeline

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

        DC_is = [edcr.learn_detection_rules(g) for g in data_preprocessing.granularities]
        CC_is = [edcr.learn_correction_rules(g) for g in data_preprocessing.granularities]

        for DC_i, CC_i in zip(DC_is, CC_is):
            vit_pipeline.run_combined_fine_tuning_pipeline(lrs=[0.0001],
                                                           loss='LTN_BCE',
                                                           DC_i=DC_i,
                                                           CC_i=CC_i)

if __name__ == '__main__':
    run()

