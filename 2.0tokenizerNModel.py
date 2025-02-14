from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

classifier = pipeline("sentiment-analysis")

result = classifier("I am the best NOT")

print(result)

# now with 
model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier_ = pipeline("sentiment-analysis", tokenizer= tokenizer, model = model)

result = classifier_("I am the best")

print(result)