from django.shortcuts import render
from django.http import JsonResponse
import joblib
import os

def predict(request):
    if request.method == 'POST':
        # Load the saved model
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model = joblib.load(os.path.join(BASE_DIR, 'insurance_model.pkl'))

        # Get input values from the form
        age = float(request.POST.get('age'))
        bmi = float(request.POST.get('bmi'))
        children = float(request.POST.get('children'))

        # Make a prediction
        prediction = model.predict([[age, bmi, children]])[0]
        return JsonResponse({'prediction': round(prediction, 2)})

    return render(request, 'predict.html')
