import gradio as gr
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

  
    
# Load the trained model
with open('best_insurance_model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_insurance_cost(age, sex, bmi, children, smoker, region):
    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
    # predict using the loaded model
    prediction = model.predict(input_data)[0]  # Get first element
    return f"Medical Insurance Cost: ${np.clip(prediction, 0, 50000):.2f}"
# Define the Gradio interface
interface = gr.Interface(
    fn=predict_insurance_cost,
    inputs=[
        gr.Number(label="Age"),
        gr.Dropdown(choices=["male", "female"], label="Sex"),
        gr.Number(label="BMI"),
        gr.Number(label="Number of Children"),
        gr.Dropdown(choices=["yes", "no"], label="Smoker"),
        gr.Dropdown(choices=["northeast", "northwest", "southeast", "southwest"], label="Region")
    ],
    outputs="text",
    title="Medical Insurance Cost Prediction",
    description="Predict the medical insurance cost based on personal attributes."
)
# Launch the interface
interface.launch(share=True)
