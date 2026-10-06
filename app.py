import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💰",
    layout="wide"
)
df = pd.read_csv("Employers_data.csv")
model = joblib.load("employee_salary_model.pkl")
group_model = joblib.load("employee_group_model.pkl")
group_scaler = joblib.load("employee_group_scaler.pkl")

# Sidebar Navigation
st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📊 Data Analysis",
        "🤖 Salary Prediction", 
        "🔍 Model Validation & Trust",
        "👥 Employee Groups"
    ]
)
if page == "🏠 Home":

    st.title("💰 Employee Salary Prediction")
    st.subheader("Machine Learning Based Salary Prediction & Analysis")

    st.write("""
    This application predicts an employee's salary based on information such as
    age, gender, department, job title, experience, education level, and location.
    """)

    st.markdown("### 📌 About the Project")

    st.write("""
    The goal of this project is to build a machine learning model that can
    predict an employee's salary using different employee-related features.
    """)

    st.markdown("### 🔎 Features Used")

    st.write("""
    - Age
    - Gender
    - Department
    - Job Title
    - Experience Years
    - Education Level
    - Location
    """)

    st.markdown("### 🤖 Machine Learning")

    st.write("""
    A Random Forest Regression model is used to predict salary.
    The model is evaluated using MAE, RMSE, R² score, and cross-validation.
    """)

    st.info(
        "💡 Use the sidebar to explore the data, analyze the model, "
        "and predict an employee's salary."
    )

elif page == "📊 Data Analysis":

    st.title("📊 Salary Data Analysis")

    st.write(
        "Explore the employee dataset and understand how different "
        "factors are related to salary."
    )

    # Dataset overview
    st.markdown("### 📋 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Employees", f"{len(df):,}")

    with col2:
        st.metric("Average Salary",f"₹{df['Salary'].mean():,.0f}")

    with col3:
        st.metric("Minimum Salary",f"₹{df['Salary'].min():,.0f}")

    with col4:
        st.metric("Maximum Salary",f"₹{df['Salary'].max():,.0f}")

    st.divider()

    # Dataset Preview
    # -------------------------

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    # Average Salary by Education
    # -------------------------

    st.subheader("Average Salary by Education Level")

    education_salary = (
        df.groupby("Education_Level")["Salary"]
        .mean()
        .sort_values()
    )

    fig = px.bar(
        education_salary,
        x=education_salary.index,
        y=education_salary.values,
        title="Average Salary by Education Level",
        labels={
            "x": "Education_Level",
            "y": "Average Salary"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    
    # st.bar_chart(education_salary)

    # Average Salary by Job Title
    # -------------------------

    st.subheader("Average Salary by Job Title")

    job_salary = (
        df.groupby("Job_Title")["Salary"]
        .mean()
        .sort_values()
    )
    fig = px.bar(
        job_salary,
        x=job_salary.index,
        y=job_salary.values,
        title="Average Salary by Job Title",
        labels={
            "x": "Job_Title",
            "y": "Average Salary"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # st.bar_chart(job_salary)

    # Salary Distribution
    # -------------------------

    st.subheader("Salary Distribution")

    fig = px.histogram(
        df,
        x="Salary",
        nbins=30,
        title="Distribution of Employee Salaries"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Salary vs Experience
    # -------------------------

    st.subheader("Salary vs Experience")

    fig = px.scatter(
        df,
        x="Experience_Years",
        y="Salary",
        title="Experience vs Salary",   
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Salary by Department
    # -------------------------

    st.subheader("Average Salary by Department")

    department_salary = (
        df.groupby("Department")["Salary"]
        .mean()
        .sort_values()
    )

    fig = px.bar(
        department_salary,
        x=department_salary.index,
        y=department_salary.values,
        title="Average Salary by Department",
        labels={
            "x": "Department",
            "y": "Average Salary"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


elif page == "🤖 Salary Prediction":

    st.title("🤖 Employee Salary Prediction")

    st.write(
        "Enter the employee details below to predict the expected salary."
    )

    st.divider()

    # INPUT FIELDS
    # -------------------------

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=65,
            value=30
        )

        gender = st.selectbox(
            "Gender",
            df["Gender"].unique()
        )

        department = st.selectbox(
            "Department",
            df["Department"].unique()
        )

        job_title = st.selectbox(
            "Job Title",
            df["Job_Title"].unique()
        )

    with col2:

        experience = st.number_input(
            "Experience (Years)",
            min_value=0,
            max_value=50,
            value=5
        )

        education = st.selectbox(
            "Education Level",
            df["Education_Level"].unique()
        )

        location = st.selectbox(
            "Location",
            df["Location"].unique()
        )

        st.divider()

    # PREDICT BUTTON
    # -------------------------

    if st.button(
        "Predict Salary",
        use_container_width=True
    ):

        input_data = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Department": [department],
            "Job_Title": [job_title],
            "Experience_Years": [experience],
            "Education_Level": [education],
            "Location": [location]
        })

        prediction = model.predict(input_data)

        predicted_salary = prediction[0]

        # Display result

        st.subheader("Prediction Result")

        st.metric(
            "Predicted Annual Salary",
            f"₹{predicted_salary:,.0f}"
        )

        # FAIR OFFER ANALYSIS
        # -------------------------
        # First compare employees with the same job title and education.
        # If fewer than 10 employees match, use the same job title only.
        matching_data = df[
            (df["Job_Title"] == job_title) &
            (df["Education_Level"] == education)
        ]

        if len(matching_data) >= 10:
            typical_salary = matching_data["Salary"].median()
            comparison_basis = "same job title and education"
        else:
            matching_data = df[df["Job_Title"] == job_title]
            typical_salary = matching_data["Salary"].median()
            comparison_basis = "same job title"

        lower_limit = typical_salary * 0.90
        upper_limit = typical_salary * 1.10

        if predicted_salary < lower_limit:
            offer_status = "Too low"
            st.warning(f"⚠️ Fair Offer: **{offer_status}**")
        elif predicted_salary > upper_limit:
            offer_status = "Too high"
            st.warning(f"⚠️ Fair Offer: **{offer_status}**")
        else:
            offer_status = "Fair"
            st.success(f"✅ Fair Offer: **{offer_status}**")

        st.metric(
            "Typical Salary",
            f"₹{typical_salary:,.0f}"
        )

        st.caption(
            f"Typical salary calculated using the median of {len(matching_data)} "
            f"matching employees ({comparison_basis})."
        )

        # st.success(
        #     f" Predicted Salary: ₹{predicted_salary:,.2f}"
        # )

        st.info(
            "The prediction is generated using the saved machine "
            "learning pipeline, including preprocessing and the "
            "trained regression model."
        )
   

elif page == "🔍 Model Validation & Trust":
    st.title(" Model Validation & Trust")
    
    st.header("Model Validation & Trust")

    st.write(
        "This section evaluates whether the model performs well "
        "and whether the high R² score is trustworthy."
    )

    # Model Metrics
    # -------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("R² Score","0.9918")

    with col2:
        st.metric("MAE","₹3,337.94")

    with col3:
        st.metric("RMSE","₹4,168.34")

    with col4:
        st.metric("MSE","17,375,021.62")

    st.divider()

    # Explanation
    # -------------------------

    st.subheader("What do these metrics mean?")

    st.write(
        """
        **R² Score:** Approximately 99.18% of the variation in salary
        is explained by the model.

        **MAE:** On average, the prediction differs from the actual
        salary by approximately ₹3,337.94.

        **RMSE:** RMSE is approximately ₹4,168.34 and gives more
        penalty to larger prediction errors.
        """
    )

    # Trust Warning
    # -------------------------

    st.warning(
        "A very high R² score does not automatically mean the model "
        "is perfect. Model performance should also be checked using "
        "MAE, RMSE, train/test comparison, cross-validation, "
        "residual analysis and leakage checks."
    )

    # Train/Test Performance
    # -------------------------

    st.subheader("Train vs Test Performance")

    validation_data = pd.DataFrame({
        "Dataset": ["Training", "Testing"],
        "R² Score": [0.991389, 0.991756]
    })

    st.dataframe(
        validation_data,
        use_container_width=True
    )

    st.success(
        "The training and testing R² scores are very close, "
        "which suggests that the model is not showing a large "
        "generalization gap."
    )

    # 5-Fold Cross-Validation
    # -------------------------

    st.subheader("5-Fold Cross-Validation")

    cv_scores = [
        0.99135658,
        0.99149756,
        0.99153940,
        0.99123652,
        0.99099503
    ]
    cv_mean = 0.991325
    cv_std = 0.000197

    cv_data = pd.DataFrame({
        "Fold": [1, 2, 3, 4, 5],
        "R² Score": cv_scores
    })

    st.dataframe(
        cv_data,
        use_container_width=True
    )
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Mean CV R²", f"{cv_mean:.6f}")

    with col2:
        st.metric("CV Std", f"{cv_std:.6f}")

    st.success(
        "The mean cross-validation R² is approximately 0.9913 "
        "with a very small standard deviation, indicating "
        "consistent performance across the five folds."
    )

    # Residual Analysis
    # -------------------------

    st.subheader("📉 Residual Analysis")

    st.write(
        """
        Residual analysis was performed in the notebook to check
        whether prediction errors showed a strong systematic pattern.

        A good residual plot should show errors scattered around
        zero without a clear pattern.
        """
    )
     # -------------------------
    # Why Trust the High R²?
    # -------------------------

    st.subheader(" Why is the high R² considered trustworthy?")

    st.write(
        """
        The R² score of 0.9918 was not evaluated in isolation.

        The model was checked using:

        • Train vs Test performance
        • 5-Fold Cross-Validation
        • MAE
        • RMSE
        • Residual analysis
        • Data leakage checks
        """
    )
    st.success(
        "The training and testing R² scores are very close, "
        "and the cross-validation scores are also consistently high. "
        "These checks provide stronger evidence that the model "
        "generalizes well to unseen data."
    )


# EMPLOYEE GROUPS PAGE
# ----------------------------------------

elif page == "👥 Employee Groups":

    st.header("Employee Groups")

    st.write(
        "Group employees based on age, experience and salary using K-Means."
    )

    st.divider()

    group_names = {
        0: "Senior High Earners",
        1: "Early Career",
        2: "Mid Career",
        3: "Experienced Professionals"
    }

    group_features = df[["Age", "Experience_Years", "Salary"]]
    scaled_group_data = group_scaler.transform(group_features)
    df_groups = df.copy()
    df_groups["Employee_Group"] = group_model.predict(scaled_group_data)
    df_groups["Group_Name"] = df_groups["Employee_Group"].map(group_names)

    st.subheader("Employees in Each Group")

    group_counts = (
        df_groups["Group_Name"]
        .value_counts()
        .reindex(group_names.values(), fill_value=0)
    )

    counts_table = group_counts.rename("Number of Employees").reset_index()
    counts_table.columns = ["Employee Group", "Number of Employees"]

    st.dataframe(counts_table, use_container_width=True, hide_index=True)

    st.subheader("Group Averages")

    group_summary = (
        df_groups.groupby("Group_Name")[["Age", "Experience_Years", "Salary"]]
        .mean()
        .reindex(group_names.values())
        .round(2)
        .reset_index()
    )

    group_summary["Salary"] = group_summary["Salary"].map(lambda x: f"₹{x:,.0f}")
    group_summary.columns = [
        "Employee Group", "Average Age", "Average Experience", "Average Salary"
    ]

    st.dataframe(group_summary, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Find an Employee's Group")
    st.write("Enter age, experience and salary to see which employee group they belong to.")

    col1, col2, col3 = st.columns(3)

    with col1:
        group_age = st.number_input("Age", min_value=18, max_value=65, value=22)

    with col2:
        group_experience = st.number_input("Experience (Years)", min_value=0, max_value=50, value=1)

    with col3:
        group_salary = st.number_input("Salary", min_value=0, value=35000, step=5000)

    if st.button("Find Employee Group", use_container_width=True):

        new_employee = np.array([[group_age, group_experience, group_salary]])
        new_employee_scaled = group_scaler.transform(new_employee)
        predicted_group = group_model.predict(new_employee_scaled)[0]
        predicted_group_name = group_names[predicted_group]

        st.success(f"Employee Group: **{predicted_group_name}**")



