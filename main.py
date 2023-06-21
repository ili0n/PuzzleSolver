import requests
import torch
import torchmetrics.functional
from PIL import Image
from transformers.models.vit.image_processing_vit import ViTImageProcessor
from transformers import ViTForImageClassification, EvalPrediction, ViTModel, ViTConfig, Trainer
from tqdm import tqdm
from evaluate import load
import numpy as np
import urllib.parse as parse
import os
from torchmetrics import ExtendedEditDistance
from datasets import load_dataset
from transformers import TrainingArguments
from torchinfo import summary
import torch.nn as nn
from Levenshtein import distance
from MLPHeadRL import CustomFFN

from MLPHead import MLPHead

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
in_channels = 3

# %%
# the model name
model_name = "google/vit-base-patch32-224-in21k"
# load the image processor
image_processor = ViTImageProcessor.from_pretrained(model_name)
#
# # loading the pre-trained model
#
#
#
#
# # a function to determine whether a string is a URL or not
# def is_url(string):
#     try:
#         result = parse.urlparse(string)
#         return all([result.scheme, result.netloc, result.path])
#     except:
#         return False
#
#
# # a function to load an image
# def load_image(image_path):
#     if is_url(image_path):
#         return Image.open(requests.get(image_path, stream=True).raw)
#     elif os.path.exists(image_path):
#         return Image.open(image_path)
#
#
# # %%
# def get_prediction(model, url_or_path):
#     # load the image
#     img = load_image(url_or_path)
#     # preprocessing the image
#     pixel_values = image_processor(img, return_tensors="pt")["pixel_values"].to(device)
#     # perform inference
#     output = model(pixel_values)
#     # get the label id and return the class name
#     return model.config.id2label[int(output.logits.softmax(dim=1).argmax())]


# load the custom dataset
ds = load_dataset("imagefolder", data_dir="puzzles")

# # Exploring the Data
labels = ds["train"].features["label"]
labels.int2str(ds["train"][532]["label"])


# # Preprocessing the Data

# %%
def transform(examples):
    # convert all images to RGB format, then preprocessing it
    # using our image processor
    inputs = image_processor([img.convert("RGB") for img in examples["image"]], return_tensors="pt")
    # we also shouldn't forget about the labels
    inputs["labels"] = examples["label"]
    return inputs


# %%
# use the with_transform() method to apply the transform to the dataset on the fly during training
dataset = ds.with_transform(transform)

# %%
for item in dataset["train"]:
    print(item["pixel_values"].shape)
    print(item["labels"])
    break

# extract the labels for our dataset
labels = ds["train"].features["label"].names


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }


# # Defining the Metrics


# load the accuracy and f1 metrics from the evaluate module
accuracy = load("accuracy")
f1 = load("f1")


def compute_metrics(eval_pred):
    # compute the accuracy and f1 scores & return them
    accuracy_score = accuracy.compute(predictions=np.argmax(eval_pred.predictions, axis=1),
                                      references=eval_pred.label_ids)
    f1_score = f1.compute(predictions=np.argmax(eval_pred.predictions, axis=1), references=eval_pred.label_ids,
                          average="macro")
    return {**accuracy_score, **f1_score}
    # score = []
    # for i in range(eval_pred.predictions):
    #     score.append(distance(eval_pred.predictions[i], eval_pred.label_ids[i]))
    # return {**accuracy_score, **score}


# # Training the Model

# %%
# load the ViT model
# model = ViTForImageClassification.from_pretrained(
#     model_name,
#     num_labels=len(labels),
#     id2label={str(i): c for i, c in enumerate(labels)},
#     label2id={c: str(i) for i, c in enumerate(labels)},
#     ignore_mismatched_sizes=True,
# )

config = ViTConfig.from_pretrained(model_name)
config.patch_size = 16
config.in_channels = in_channels
config.label2id = {str(i): c for i, c in enumerate(labels)}
config.id2label = {str(i): c for i, c in enumerate(labels)}
model = ViTForImageClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)

grad_count = 0

for name, param in model.named_parameters():
    if grad_count == 0:
        grad_count+=1
        continue
    if param.requires_grad:
        param.requires_grad_(False)

print("Grad count: " + str( grad_count))


input_dim = 768
output_dim = 16  # Number of classes in the classification task
summary(model, (32, 3, 224, 224))

# for name, module in model.named_modules():
#     if isinstance(module,torch.nn.Module):
#         print(name)
#
# # for param in model.parameters():
# #     param.requires_grad = False
#
# # Create the new MLP head
# mlp_head = MLPHead(input_dim, output_dim)
model.mlp_head = CustomFFN(config.hidden_size,config.patch_size)

# model.head = mlp_head
model.to(device)
for name, module in model.named_modules():
    if isinstance(module,torch.nn.Module):
        print(name)

# model = ViTForImageClassification.from_pretrained(f"./vit-base-catjig/checkpoint-850").to(device)

training_args = TrainingArguments(
    output_dir="./vit-base-catjig",  # output directory
    per_device_train_batch_size=32,  # batch size per device during training
    evaluation_strategy="steps",  # evaluation strategy to adopt during training
    num_train_epochs=40,  # total number of training epochs
    # fp16=True,                    # use mixed precision
    # save_steps=1000,  # number of update steps before saving checkpoint
    # eval_steps=1000,  # number of update steps before evaluating
    # logging_steps=1000,  # number of update steps before logging
    save_steps=1000,
    eval_steps=1000,
    logging_steps=200,
    save_total_limit=2,  # limit the total amount of checkpoints on disk
    remove_unused_columns=False,  # remove unused columns from the dataset
    push_to_hub=False,  # do not push the model to the hub
    report_to='tensorboard',  # report metrics to tensorboard
    load_best_model_at_end=True,  # load the best model at the end of training

)

# %%


trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    data_collator=collate_fn,  # the data collator that will be used for batching
    compute_metrics=compute_metrics,  # the metrics function that will be used for evaluation
    train_dataset=dataset["train"],  # training dataset
    eval_dataset=dataset["validation"],  # evaluation dataset
    tokenizer=image_processor,  # the processor that will be used for preprocessing the images
)
print("training")
# %%
# start training
trainer.train()

# %%
trainer.evaluate()




# trainer.evaluate(dataset["test"])


# %%
# start tensorboard
# %load_ext tensorboard
# %reload_ext
# tensorboard
# %tensorboard - -logdir. / vit - base - food / runs
# %%
# example 1
# get_prediction_probs(model,
#                      "https://images.pexels.com/photos/406152/pexels-photo-406152.jpeg?auto=compress&cs=tinysrgb&w=600")
#
# # %%
# # example 2
# get_prediction_probs(model,
#                      "https://images.pexels.com/photos/920220/pexels-photo-920220.jpeg?auto=compress&cs=tinysrgb&w=600")
#
# # %%
# # example 3
# get_prediction_probs(model,
#                      "https://images.pexels.com/photos/3338681/pexels-photo-3338681.jpeg?auto=compress&cs=tinysrgb&w=600")
#
# # %%
# # example 4
# get_prediction_probs(model,
#                      "https://images.pexels.com/photos/806457/pexels-photo-806457.jpeg?auto=compress&cs=tinysrgb&w=600",
#                      num_classes=10)
#
# # %%
# get_prediction_probs(model,
#                      "https://images.pexels.com/photos/1624487/pexels-photo-1624487.jpeg?auto=compress&cs=tinysrgb&w=600")
