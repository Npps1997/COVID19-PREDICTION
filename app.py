import streamlit as st
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Define Streamlit app
st.title('Covid19 Test Prediction')

# Home page
def home():
    st.header('Welcome to the Covid19 Prediction Portal.')
    
    # Image
    st.image('pic11.jpg', width=800)

    # Intro
    st.write("""
        This portal provides a simple and intuitive interface to predict the likelihood of COVID-19 based on user input.
        Please use the left-side navigation dropdown to explore different sections of the application.
    """)
        

    # requirements
    st.markdown('---')
    st.subheader('Latest News and Updates')
    st.write("""
        Stay informed with the latest news and updates related to COVID-19.
        Check out the official [WHO website](https://www.who.int/) for reliable information.
    """)

    # social media links or other relevant information
    st.markdown('---')
    st.subheader('Connect with Us')
    st.write("""
        Follow us on social media for the latest updates and community discussions:
        - [linkedin](https://www.linkedin.com/in/np-229013245)
        - [Github](https://github.com/Npps1997)
    """)



# Prediction page
def predict_datapoint():
    st.header('Covid19 Details form:')
    # Create form for user input
    cough_symptoms = st.selectbox('Cough Symptoms', ['', 'True', 'False'], format_func=lambda x: 'Choose an option' if x == '' else x)
    fever = st.selectbox('Fever', ['', 'True', 'False'], format_func=lambda x: 'Choose an option' if x == '' else x)
    sore_throat = st.selectbox('Sore Throat', ['', 'True', 'False'], format_func=lambda x: 'Choose an option' if x == '' else x)
    shortness_of_breath = st.selectbox('Shortness of Breath', ['', 'True', 'False'], format_func=lambda x: 'Choose an option' if x == '' else x)
    headache = st.selectbox('Headache', ['', 'True', 'False'], format_func=lambda x: 'Choose an option' if x == '' else x)
    age_60_above = st.selectbox('Age 60 Above', ['', 'Yes', 'No', 'Unspecified'], format_func=lambda x: 'Choose an option' if x == '' else x)
    sex = st.selectbox('Sex', ['', 'Male', 'Female'], format_func=lambda x: 'Choose an option' if x == '' else x)
    known_contact = st.selectbox('Known Contact', ['', 'Abroad', 'Other', 'Contact with confirmed'], format_func=lambda x: 'Choose an option' if x == '' else x)

    # Check if the form is submitted
    if st.button('Predict Corona'):
        # Check if the user made a valid selection before proceeding
        if '' not in [cough_symptoms, fever, sore_throat, shortness_of_breath, headache, age_60_above, sex, known_contact]:
            data = CustomData(
                Cough_symptoms=(cough_symptoms == 'True'),
                Fever=(fever == 'True'),
                Sore_throat=(sore_throat == 'True'),
                Shortness_of_breath=(shortness_of_breath == 'True'),
                Headache=(headache == 'True'),
                Age_60_above=age_60_above,
                Sex=sex.lower(),
                Known_contact=known_contact
            )

            # st.write("Before Prediction")

            predict_pipeline = PredictPipeline()
            # st.write("Mid Prediction")
            results = predict_pipeline.predict(data.get_data_as_data_frame())
            # st.write("After Prediction")

            if results[0] == 1:
                result_text = "Positive"
            else:
                result_text = "Negative"

            st.write(f"Prediction: {result_text}")
        else:
            st.warning("Please choose an option for each input before predicting.")

# main Streamlit app
def main():
    menu = ['Home', 'Covid19 Prediction']
    choice = st.sidebar.selectbox('Navigation', menu)

    if choice == 'Home':
        home()
    elif choice == 'Covid19 Prediction':
        predict_datapoint()

if __name__ == '__main__':
    main()
