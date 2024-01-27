import vit_pipeline
import EDCR_pipeline
import utils


def run():
    mults = [False, True]
    lrs = [0.0001, 3e-06]
    losses = ['BCE', 'soft_marginal']
    from_main = [True, False]
    combined = [False, True]

    for comb in combined:
        for main in from_main:
            for mult in mults:
                for loss in losses:
                    for lr in lrs:
                        if not comb and lr == 3e-06:
                            continue

                        EDCR_pipeline.run_EDCR_pipeline(
                            main_lr=lr,
                            combined=comb,
                            loss=loss,
                            conditions_from_secondary=not main,
                            conditions_from_main=main,
                            consistency_constraints=True,
                            multiprocessing=mult)