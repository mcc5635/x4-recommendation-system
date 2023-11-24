import streamlit as st
import pandas as pd


@st.cache
def load_and_preprocess_data():
    df = pd.read_csv(
        "Data/OnlineRetail.csv",
        encoding="latin-1",
    )

    # Remove nans values
    df = df.dropna()

    # Use only positive quantites. This is not a robust approach,
    # but to keep things simple it quite good.
    df = df[df["Quantity"] > 0]

    # Parse the date column and add 10 years, just to better visualization
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"]).dt.floor(
        "d"
    ) + pd.offsets.DateOffset(years=10)

    # Change customer id to int
    df["CustomerID"] = df["CustomerID"].astype(int)

    # Add price column
    df["Price"] = df["Quantity"] * df["UnitPrice"]

    # Get unique entries in the dataset of users and products
    users = df["CustomerID"].unique()
    products = df["StockCode"].unique()

    # Create a categorical type for users and product. User ordered to ensure
    # reproducibility
    user_cat = pd.CategoricalDtype(categories=sorted(users), ordered=True)
    product_cat = pd.CategoricalDtype(categories=sorted(products), ordered=True)

    # Transform and get the indexes of the columns
    user_idx = df["CustomerID"].astype(user_cat).cat.codes
    product_idx = df["StockCode"].astype(product_cat).cat.codes

    # Add the categorical index to the starting dataframe
    df["CustomerIndex"] = user_idx
    df["ProductIndex"] = product_idx

    return df, user_idx, product_idx