from transformers import pipeline

classifier = pipeline("zero-shot-classification")

result = classifier(
    "I am here because I am with God",
    cadidate_labels = ["education", "politics", "bussiness"])

print(result);