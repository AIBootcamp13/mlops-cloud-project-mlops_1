from modeling.src.trainer.trainer import Trainer

class BaselineTrainer(Trainer):
    def __init__(self, model_name, epochs, batch_size, outputs, scaler, window_size):
        super().__init__(model_name, epochs, batch_size, outputs, scaler, window_size)
        print('Baseline Trainer')

    def split_data(self, data):
        super().split_data_common(data)

    def train_model(self):
        return super().train_model_common()
    
    def save_model(self, model_root_path, save_model_name, is_store_in_s3=False):
        super().save_model_common(model_root_path, save_model_name, is_store_in_s3)
