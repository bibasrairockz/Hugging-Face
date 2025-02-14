from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier_ = pipeline("sentiment-analysis", tokenizer= tokenizer, model = model)

result = classifier_("I am the best")

print(result)

string = "It is a lovely weather outside"
print(string)
print(tokenizer(string))
tokens = tokenizer.tokenize(string)
print(tokens)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
decoded_string = tokenizer.decode(ids)
print(decoded_string)
