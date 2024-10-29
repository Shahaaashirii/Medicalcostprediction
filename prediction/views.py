
from django.shortcuts import render
import joblib
import numpy as np

def predict_charge(request):
    if request.method == 'POST':
        # Get data from form
        age = float(request.POST.get('age'))
        sex = request.POST.get('sex')  # female = 1, male = 0
        bmi = float(request.POST.get('bmi'))
        children = int(request.POST.get('children'))
        smoker = request.POST.get('smoker')  # yes = 1, no = 0
        region = request.POST.get('region')  # Handle this according to your encoding

        # Convert sex and smoker to binary format
        sex = 1 if sex == 'female' else 0
        smoker = 1 if smoker == 'yes' else 0

        # Handle region encoding (this must match the encoding during model training)
        regions = ['northwest', 'southeast', 'southwest']
        region_encoded = [1 if region == r else 0 for r in regions]

        # Prepare the input array for prediction
        input_data = [[age, sex, bmi, children, smoker] + region_encoded]

        # Load the scaler and apply transformation
        scaler = joblib.load('prediction/model/scaler.pkl')
        
        input_data_scaled = scaler.transform(input_data)

        # Load the saved model
        model = joblib.load('prediction/model/insurance_model.pkl')

        # Make the prediction
        predicted_charge = model.predict(input_data_scaled)[0]

        # Render the result on the same page
        return render(request, 'prediction/index.html', {'prediction': round(predicted_charge, 2)})

    return render(request, 'prediction/index.html')
