from transformers import Trainer

from CustomLossNN import CustomLossNN


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs,return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0]
        my_custom_loss = CustomLossNN()
        my_custom_loss.to(device=model.device)
        loss = my_custom_loss(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
