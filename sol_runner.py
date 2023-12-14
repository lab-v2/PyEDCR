import vit_pipeline
import EDCR_pipeline


def run():
    vit_pipeline.run_combined_fine_tuning_pipeline()
    EDCR_pipeline.run_EDCR()
