import vit_pipeline
import EDCR_pipeline
import utils


def run():
    EDCR_pipeline.run_EDCR_pipeline(
        main_lr=1e-06,
        combined=False,
        loss='BCE',
        conditions_from_secondary=True,
        conditions_from_main=False,
        consistency_constraints=False,
        multiprocessing=True)

    EDCR_pipeline.run_EDCR_pipeline(
        main_lr=0.0001,
        combined=False,
        loss='BCE',
        conditions_from_secondary=True,
        conditions_from_main=False,
        consistency_constraints=False,
        multiprocessing=True)

    EDCR_pipeline.run_EDCR_pipeline(
        main_lr=3e-06,
        combined=True,
        loss='BCE',
        conditions_from_secondary=True,
        conditions_from_main=False,
        consistency_constraints=False,
        multiprocessing=True)

    EDCR_pipeline.run_EDCR_pipeline(
        main_lr=0.0001,
        combined=True,
        loss='BCE',
        conditions_from_secondary=True,
        conditions_from_main=False,
        consistency_constraints=False,
        multiprocessing=True)

    EDCR_pipeline.run_EDCR_pipeline(
        main_lr=3e-06,
        combined=True,
        loss='soft_marginal',
        conditions_from_secondary=True,
        conditions_from_main=False,
        consistency_constraints=False,
        multiprocessing=True)

    EDCR_pipeline.run_EDCR_pipeline(
        main_lr=0.0001,
        combined=True,
        loss='soft_marginal',
        conditions_from_secondary=True,
        conditions_from_main=False,
        consistency_constraints=False,
        multiprocessing=True)