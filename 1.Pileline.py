from transformers import pipeline

classifier = pipeline("sentiment-analysis")

result = classifier("I have the best disadvantage")

print(result)