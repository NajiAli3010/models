# from transformers import T5ForConditionalGeneration, T5Tokenizer
# import torch

# # Step 1: Load the trained model and tokenizer
# model_path = './trained_model_1'
# model = T5ForConditionalGeneration.from_pretrained(model_path)
# tokenizer = T5Tokenizer.from_pretrained(model_path)


# # Step 2: Function to predict components based on dish name
# def predict_components_and_type(dish_name):
#     input_text = "predict ingredients and type: " + dish_name
#     input_ids = tokenizer.encode(input_text, return_tensors="pt")

#     with torch.no_grad():
#         outputs = model.generate(input_ids, max_length=512, num_beams=5, early_stopping=True)

#     predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return predicted_text


# if __name__ == "__main__":
#     while True:
#         dish_name = input("Enter a dish name (or type 'exit' to quit): ")
#         if dish_name.lower() == 'exit':
#             break
#         prediction = predict_components_and_type(dish_name)
#         print(f"Prediction for '{dish_name}': {prediction}\n")

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from transformers import T5ForConditionalGeneration, T5Tokenizer
from fastapi.middleware.cors import CORSMiddleware
import torch

app = FastAPI()

# Step 1: Load the trained model and tokenizer
model_path = './trained_model_1'
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Step 2: Function to predict components based on dish name
def predict_components_and_type(dish_name):
    input_text = "predict ingredients and type: " + dish_name
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=512, num_beams=5, early_stopping=True)

    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return predicted_text


class DishInput(BaseModel):
    dish_names: List[str]

origins = [
"http://localhost",
"http://localhost:3000",  # Example frontend URL
# Add other frontend URLs as needed
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(dish_input: DishInput):
    print(dish_input)
    predictions = {}
    for dish_name in dish_input.dish_names:
        dish_name = dish_name.strip()
        if dish_name:
            prediction = predict_components_and_type(dish_name)
            predictions[dish_name] = prediction
    return predictions

if __name__ == "__main__":
    while True:
        dish_names = input("Enter dish names separated by a comma (or type 'exit' to quit): ")
        if dish_names.lower() == 'exit':
            break
        dish_names = dish_names.split(',')
        for dish_name in dish_names:
            dish_name = dish_name.strip()  # Remove leading/trailing whitespace
            if dish_name:  # Check if the string is not empty
                prediction = predict_components_and_type(dish_name)
                print(f"Prediction for '{dish_name}': {prediction}\n")
