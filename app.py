import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import time


st.set_page_config(
    page_title="Credit Score Predictor",
    page_icon="💳",
    layout="wide"
)


MODEL_FILE = 'best_credit_score_model.joblib'
IMPUTER_FILE = 'imputer.joblib'
SCALER_FILE = 'scaler.joblib'
DATA_FILE = 'cs-test-modified.csv'


@st.cache_resource
def load_model_and_preprocessors():
    """Loads the ML model, imputer, and scaler from disk. Caches for performance."""
    if not all(os.path.exists(f) for f in [MODEL_FILE, IMPUTER_FILE, SCALER_FILE]):
        st.error("Model files not found! Please ensure 'best_credit_score_model.joblib', 'imputer.joblib', and 'scaler.joblib' are in the same directory.")
        return None, None, None
    model = joblib.load(MODEL_FILE)
    imputer = joblib.load(IMPUTER_FILE)
    scaler = joblib.load(SCALER_FILE)
    return model, imputer, scaler

@st.cache_data
def load_and_prepare_dataframe(file_path):
    """Loads and prepares the comparison dataframe. Caches for performance."""
    if not os.path.exists(file_path):
        st.error(f"Data file not found! Please ensure '{file_path}' is in the directory.")
        return None
        
    df = pd.read_csv(file_path)

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    median_income = df['MonthlyIncome'].median()
    df['MonthlyIncome'].fillna(median_income, inplace=True)
    mode_dependents = df['NumberOfDependents'].mode()[0]
    df['NumberOfDependents'].fillna(mode_dependents, inplace=True)

    age_bins = [20, 30, 40, 50, 60, 70, 80, 90, 110]
    age_labels = ['21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91+']
    df['AgeGroup'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

    try:
        df['IncomeGroup'] = pd.qcut(df['MonthlyIncome'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
    except ValueError:
        income_bins = [0, 2500, 5000, 7500, 10000, np.inf]
        income_labels = ['0-2500', '2501-5000', '5001-7500', '7501-10000', '10001+']
        df['IncomeGroup'] = pd.cut(df['MonthlyIncome'], bins=income_bins, labels=income_labels, right=False)
        
    return df


model, imputer, scaler = load_model_and_preprocessors()
df = load_and_prepare_dataframe(DATA_FILE)

st.title("💳 Credit Score & Risk Predictor")
st.markdown("Enter your financial details below to estimate your credit score and see how your default risk compares to others in your age and income group.")


with st.expander("Enter Your Financial Information", expanded=True):
    with st.form(key='credit_score_form'):
        
  
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=110, value=45, step=1)
            monthly_income = st.number_input("Monthly Income ($)", min_value=0, value=6000, step=100)
            dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=2, step=1)
            open_credit_lines = st.number_input("Open Credit Lines and Loans", min_value=0, value=13, step=1)
        
        with col2:
            past_due_30_59 = st.number_input("Times 30-59 Days Past Due", min_value=0, value=2, step=1)
            past_due_60_89 = st.number_input("Times 60-89 Days Past Due", min_value=0, value=0, step=1)
            past_due_90 = st.number_input("Times 90+ Days Late", min_value=0, value=0, step=1)
            real_estate_loans = st.number_input("Real Estate Loans or Lines", min_value=0, value=6, step=1)

        with col3:
            utilization = st.slider("Revolving Utilization of Unsecured Lines", 0.0, 2.0, 0.76, 0.01, help="Total balance on credit cards and personal lines of credit divided by the sum of credit limits.")
            debt_ratio = st.slider("Debt Ratio", 0.0, 5.0, 0.8, 0.01, help="Monthly debt payments, alimony, living costs divided by monthly gross income.")
        

        predict_button = st.form_submit_button(label='Predict My Score', type="primary", use_container_width=True)



if predict_button and model is not None:

    user_input = {
        'RevolvingUtilizationOfUnsecuredLines': utilization,
        'age': age,
        'NumberOfTime30-59DaysPastDueNotWorse': past_due_30_59,
        'DebtRatio': debt_ratio,
        'MonthlyIncome': monthly_income,
        'NumberOfOpenCreditLinesAndLoans': open_credit_lines,
        'NumberOfTimes90DaysLate': past_due_90,
        'NumberRealEstateLoansOrLines': real_estate_loans,
        'NumberOfTime60-89DaysPastDueNotWorse': past_due_60_89,
        'NumberOfDependents': dependents
    }

    input_df = pd.DataFrame([user_input])
    input_features = imputer.transform(input_df)
    input_scaled = scaler.transform(input_features)

    probability_of_default = model.predict_proba(input_scaled)[:, 1][0]
    credit_score = 350 + (850 - 350) * (1 - probability_of_default)
    

    st.markdown("<a id='results_section'></a>", unsafe_allow_html=True)
    st.markdown("---")
    st.header("Prediction Results")
    
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric(label="Estimated Credit Score", value=f"{int(credit_score)}")
    with res_col2:
        st.metric(label="Probability of Default", value=f"{probability_of_default:.2%}")


    if df is not None:
        st.header("How You Compare")

        age_bins = [20, 30, 40, 50, 60, 70, 80, 90, 110]
        age_labels = ['21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91+']
        user_age_group = pd.cut([age], bins=age_bins, labels=age_labels, right=False)[0]

        try:
            _, income_bins_qcut = pd.qcut(df['MonthlyIncome'], q=4, retbins=True, duplicates='drop')
            user_income_group = pd.cut([monthly_income], bins=income_bins_qcut, labels=['Low', 'Medium', 'High', 'Very High'], right=True, include_lowest=True)[0]
        except (ValueError, IndexError):
            income_bins = [0, 2500, 5000, 7500, 10000, np.inf]
            income_labels = ['0-2500', '2501-5000', '5001-7500', '7501-10000', '10001+']
            user_income_group = pd.cut([monthly_income], bins=income_bins, labels=income_labels, right=False)[0]

        filtered_df = df[(df['AgeGroup'] == user_age_group) & (df['IncomeGroup'] == user_income_group)]

        if not filtered_df.empty:
            
            plot_col, stats_col = st.columns([2, 1]) 
            
            with plot_col:

                fig, ax = plt.subplots()
                
                ax.boxplot(filtered_df['Probability'], patch_artist=True,
                           boxprops=dict(facecolor='lightblue', color='blue'),
                           medianprops=dict(color='orange', linewidth=2))
                ax.plot(1, probability_of_default, 'r*', markersize=15, label='Your Probability')
                
                # Text and label colors
                ax.set_title(f'Default Probability Distribution\nAge: {user_age_group} & Income: {user_income_group}')
                ax.set_ylabel('Probability')
                ax.set_xticklabels([''])
                ax.legend()
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with stats_col:
                st.subheader("Group Statistics")
                st.write(f"Stats for the **{filtered_df.shape[0]}** people in your comparison group:")
                st.dataframe(filtered_df['Probability'].describe())
        else:
            st.warning("No comparison data found for your specific age and income group.")
            
    st.components.v1.html(
        """
        <script>
            window.parent.document.getElementById('results_section').scrollIntoView({behavior: 'smooth'});
        </script>
        """,
        height=0
    )


elif predict_button and model is None:
    st.error("Cannot predict because the model files could not be loaded. Please check the console for errors.")
