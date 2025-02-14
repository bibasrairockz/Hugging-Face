from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier_ = pipeline("sentiment-analysis", tokenizer= tokenizer, model = model)

X_train = ["I have so much to say", "But you are not ready to listen"]

result = classifier_(X_train)

print(result)

# now saving
path = "./saved"
tokenizer.save_pretrained(path)
model.save_pretrained(path)

t = AutoTokenizer.from_pretrained(path)
m = AutoModelForSequenceClassification.from_pretrained(path)

classifier_ = pipeline("sentiment-analysis", tokenizer= t, model = m)

X_train = ["I have so much to say", "But you are not ready to listen"]

result = classifier_(X_train)

print(result)