from transformers import Trainer

import CustomLoss


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0]
        my_custom_loss = CustomLoss.CustomLoss()
        return my_custom_loss(logits, labels)
