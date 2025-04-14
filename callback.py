from pytorch_lightning.callbacks import Callback

class MyPrintCallback(Callback):

    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print('Training started.')

    def on_train_end(self, trainer, pl_module):
        print('Training ended.')

    def on_validation_start(self, trainer, pl_module):
        print('Validation started.')

    def on_validation_end(self, trainer, pl_module):
        print('Validation ended.')

    def on_test_start(self, trainer, pl_module):
        print('Test started.')

    def on_test_end(self, trainer, pl_module):
        print('Test ended.')