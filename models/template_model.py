from base.base_model import BaseModel
import tensorflow as tf


class TemplateModel(BaseModel):
    def __init__(self, config):
        super(TemplateModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        # here you build the tensorflow graph of any model you want and also define the loss.
        pass
