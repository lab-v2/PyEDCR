class ExperimentConfig:
    def __init__(self,
                 data_str: str,
                 main_model_name: str,
                 main_lr: float,
                 original_num_epochs: int,
                 secondary_model_name: str = None,
                 secondary_num_epochs: int = None,
                 secondary_lr: float = None,
                 binary_model_name: str = None,
                 binary_num_epochs: int = None,
                 binary_lr: float = None):
        self.data_str = data_str
        self.main_model_name = main_model_name
        self.main_lr = main_lr
        self.original_num_epochs = original_num_epochs
        self.secondary_model_name = secondary_model_name
        self.secondary_lr = secondary_lr
        self.secondary_num_epochs = secondary_num_epochs
        self.binary_model_name = binary_model_name
        self.binary_lr = binary_lr
        self.binary_num_epochs = binary_num_epochs
