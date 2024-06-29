import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


lr_model = joblib.load('lr_bmi_predictor.pkl')
rf_model = joblib.load('rf_bmi_predictor.pkl')
metrics = joblib.load('metrics.pkl')


st.title('BMI Predictor')



height = st.number_input('Height (m)', min_value=0.0, step=0.01)
weight = st.number_input('Weight (kg)', min_value=0.0, step=0.1)

model_option = st.selectbox(
    'Select Model',
    ('Linear Regression', 'Random Forest')
)

st.write("### Model Metrics")
if model_option == 'Linear Regression':
    st.write(f"**Mean Absolute Error (MAE):** {metrics['Linear Regression']['MAE']:.2f}")
    st.write(f"**R² Score:** {metrics['Linear Regression']['R2']:.2f}")
elif model_option == 'Random Forest':
    st.write(f"**Mean Absolute Error (MAE):** {metrics['Random Forest']['MAE']:.2f}")
    st.write(f"**R² Score:** {metrics['Random Forest']['R2']:.2f}")


if st.button('Predict'):
    if height > 0 and weight > 0:
        if model_option == 'Linear Regression':
            bmi = lr_model.predict([[height, weight]])[0]
        elif model_option == 'Random Forest':
            bmi = rf_model.predict([[height, weight]])[0]
        
        st.success(f'Your predicted BMI is {bmi:.2f}')
        
       
        print(f"Height: {height}, Weight: {weight}, Model: {model_option}, Predicted BMI: {bmi:.2f}")
    else:
        st.error('Please enter valid height and weight.')


data = pd.read_csv('bmi_data.csv')

st.write("### Model Predictions vs Actual BMI")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.scatterplot(x=data['BMI'], y=lr_model.predict(data[['Height', 'Weight']]), ax=axes[0])
axes[0].set_title('Linear Regression Predictions')
axes[0].set_xlabel('Actual BMI')
axes[0].set_ylabel('Predicted BMI')

sns.scatterplot(x=data['BMI'], y=rf_model.predict(data[['Height', 'Weight']]), ax=axes[1])
axes[1].set_title('Random Forest Predictions')
axes[1].set_xlabel('Actual BMI')
axes[1].set_ylabel('Predicted BMI')

st.pyplot(fig)
