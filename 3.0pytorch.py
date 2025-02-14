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

# now with pytorch

batch = tokenizer(X_train, padding= True, truncation= True, max_length= 512, return_tensors= 'pt')
print(batch)

with torch.no_grad():
    output = model(**batch)
    print(output)
    predictions = F.softmax(output.logits, dim= 1)
    print(predictions)
    labels = torch.argmax(predictions, dim= 1)
    print(labels)