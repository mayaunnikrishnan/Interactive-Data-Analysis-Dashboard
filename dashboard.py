import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# Function to load the dataset
def load_data():
    dataset_source = st.radio("Select dataset source", ("Upload your own dataset", "Choose from seaborn"))
    
    if dataset_source == "Upload your own dataset":
        uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx"])
        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
        else:
            st.warning("Please upload a file.")
            return None
    else:
        dataset_name = st.selectbox("Select a dataset from seaborn", sns.get_dataset_names())
        df = sns.load_dataset(dataset_name)
        
    return df
# Function to apply filters to the dataset
def filter_data(df):
    st.write("### Filter Data")
    filter_cols = st.multiselect("Select columns to filter by", df.columns.tolist())

    if filter_cols:
        for col in filter_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val, max_val = st.slider(f"Filter by {col}:", float(df[col].min()), float(df[col].max()), (float(df[col].min()), float(df[col].max())))
                df = df[(df[col] >= min_val) & (df[col] <= max_val)]
            else:
                unique_vals = df[col].unique().tolist()
                selected_vals = st.multiselect(f"Filter by {col}:", unique_vals, default=unique_vals)
                df = df[df[col].isin(selected_vals)]
    
    st.write("### Filtered Dataset")
    st.dataframe(df)

    return df


# Function to get numerical and categorical columns
def get_column_types(df):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return num_cols, cat_cols

# Function to plot heatmap
def plot_heatmap(df, num_cols, cat_cols):
    # Create a copy of the DataFrame to avoid modifying the original data
    df_copy = df.copy()
    
    # Apply label encoding to categorical columns with <= 10 categories
    enc = LabelEncoder()
    for col in cat_cols:
        if df[col].nunique() <= 10:
            df_copy[col] = enc.fit_transform(df[col])
        else:
            # Exclude this column from the heatmap
            cat_cols.remove(col)

    # Generate a list of columns for the heatmap (only numerical columns and encoded categorical columns)
    all_cols = num_cols + cat_cols
    
    # Ensure that we have numerical columns to plot
    numeric_cols = [col for col in all_cols if pd.api.types.is_numeric_dtype(df_copy[col])]
    
    if len(numeric_cols) == 0:
        st.warning("No numerical columns available for the heatmap.")
        return

    st.subheader("Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df_copy[numeric_cols].corr()  # Calculate correlation matrix only for numerical columns
    sns.heatmap(corr_matrix, annot=True, ax=ax, cmap='coolwarm')
    st.pyplot(fig)

# Function to plot pairplot
def plot_pairplot(df, num_cols):
    st.subheader("Pairplot")
    if len(num_cols) > 10:
        selected_cols = st.multiselect("Select columns for pair plot (up to 10)", num_cols, num_cols[:10])
    else:
        selected_cols = num_cols
        
    if len(selected_cols) > 0:
        pairplot_fig = sns.pairplot(df[selected_cols])
        st.pyplot(pairplot_fig.fig)

# Function to plot histogram
def plot_histogram(df, num_cols):
    st.subheader("Histogram")
    selected_col = st.selectbox("Select a column for histogram", num_cols)
    fig, ax = plt.subplots()
    sns.histplot(df[selected_col], kde=True, ax=ax)
    st.pyplot(fig)

# Function to plot pie chart
def plot_pie_chart(df, cat_cols):
    st.subheader("Pie Chart")
    selected_col = st.selectbox("Select a column for pie chart", cat_cols)
    pie_data = df[selected_col].value_counts().reset_index()
    pie_data.columns = [selected_col, 'Count']
    fig = px.pie(pie_data, names=selected_col, values='Count')
    st.plotly_chart(fig)

# Function to plot scatter or line plot
def plot_scatter_line(df, num_cols):
    st.subheader("Scatter or Line Plot")
    plot_type = st.selectbox("Select plot type", ["Scatter", "Line"])
    x_col = st.selectbox("Select X-axis column", num_cols)
    y_col = st.selectbox("Select Y-axis column", num_cols)
    
    if plot_type == "Scatter":
        fig = px.scatter(df, x=x_col, y=y_col)
    else:
        fig = px.line(df, x=x_col, y=y_col)
    
    st.plotly_chart(fig)

# Function to plot box or violin plot
def plot_box_violin(df, num_cols, cat_cols):
    st.subheader("Box or Violin Plot")
    plot_type = st.selectbox("Select plot type", ["Box", "Violin"])
    x_col = st.selectbox("Select X-axis column (categorical)", cat_cols)
    y_col = st.selectbox("Select Y-axis column (numerical)", num_cols)
    
    if plot_type == "Box":
        fig = px.box(df, x=x_col, y=y_col)
    else:
        fig = px.violin(df, x=x_col, y=y_col)
    
    st.plotly_chart(fig)

def main():
    st.title("Interactive Data Analysis Dashboard")
    
    # Step 1: Load the dataset
    df = load_data()
    
    if df is not None:
        st.write("### Dataset Overview")
        st.dataframe(df.head())

        # Step 2: Apply filters to the dataset
        filtered_df = filter_data(df)
        
        # Step 3: Proceed with plotting using the filtered data
        num_cols, cat_cols = get_column_types(filtered_df)
        
        if num_cols:
            plot_heatmap(filtered_df, num_cols, cat_cols)
            plot_pairplot(filtered_df, num_cols)
            plot_histogram(filtered_df, num_cols)
            plot_scatter_line(filtered_df, num_cols)
            if cat_cols:
                plot_box_violin(filtered_df, num_cols, cat_cols)

        if cat_cols:
            plot_pie_chart(filtered_df, cat_cols)

if __name__ == "__main__":
    main()

