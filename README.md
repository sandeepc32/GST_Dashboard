# 📊 GST Revenue Forecasting Dashboard

A Python-based interactive dashboard for GST revenue analytics, historical data exploration, and revenue forecasting using Machine Learning.

## 🚀 Project Overview

The GST Revenue Forecasting Dashboard is designed to analyze historical GST collection data, explore relationships between tax variables, and generate future revenue forecasts using Linear Regression.

The application provides an interactive interface built with Streamlit, allowing users to explore datasets, visualize trends, and evaluate forecasting results.

## ✨ Features

- **Overview:** Interactive dashboard for exploring GST revenue information.
- **Dataset Explorer:** Load Excel workbooks and select individual worksheets.
- **Revenue Analysis:** Explore historical GST trends and relationships between tax variables.
- **Revenue Prediction:** Apply Linear Regression to forecast GST revenue.
- **Model Evaluation:** Calculate regression performance metrics including R² Score, Mean Absolute Error (MAE), and Mean Squared Error (MSE).
- **Interactive Visualizations:** Analyze data through Plotly, Matplotlib, and Seaborn.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive dashboard |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Machine Learning |
| Plotly | Interactive visualizations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Excel | Dataset storage and input |

## 🧠 Machine Learning Approach

The project uses **Linear Regression**, a supervised machine learning algorithm, for GST revenue forecasting.

The workflow involves:

1. Loading historical GST data from Excel workbooks.
2. Preparing and processing the dataset.
3. Exploring historical trends and relationships between tax variables.
4. Applying Linear Regression for revenue prediction.
5. Evaluating model performance using regression metrics.
6. Presenting analytical results through interactive dashboard visualizations.

### Model Evaluation

The application imports the following evaluation metrics:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)

These metrics help assess the model's performance against available historical data.

## 📂 Project Structure

```text
GST_Dashboard/
│
├── app.py
├── app_old.py
├── app_old1.py
├── app_old2.py
├── app_old3_latest.py
├── app_old4.py
│
├── components/
│
├── modules/
│   ├── __init__.py
│   └── data_loader.py
│
├── data/
│
├── reports/
│
├── images/
│
├── sidebar.py
├── style.py
├── .gitignore
└── README.md
```

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/sandeepc32/GST_Dashboard.git
```

### 2. Navigate to the project directory

```bash
cd GST_Dashboard
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn plotly matplotlib seaborn openpyxl
```

### 5. Run the dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser, usually at:

```text
http://localhost:8501
```

## 📊 Application Modules

| Module | Description |
|---|---|
| Overview | GST revenue dashboard overview |
| Dataset | Excel dataset exploration |
| Analysis | Historical revenue analysis |
| Prediction | Machine Learning-based forecasting |

## 🎯 Project Objective

The objective of this project is to explore how historical GST collection data and machine learning techniques can support revenue analysis and forecasting.

It demonstrates the practical application of Python, data analytics, machine learning, and interactive dashboard development.

## ⚠️ Disclaimer

This project is developed for academic and analytical purposes. Forecasts are model-generated estimates based on the available input data and should not be interpreted as official GST revenue projections.

## 👨‍💻 Author

**Sandeep**

GitHub: [@sandeepc32](https://github.com/sandeepc32)

---

⭐ If you find this project interesting, consider giving the repository a star!
