from transformers import pipeline

generator = pipeline("text-generation", model= "distilgpt2")

result = generator(
    "My name is",
    max_length = 30,
    num_return_sequences = 2)

print(result)