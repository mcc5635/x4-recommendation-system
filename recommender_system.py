import streamlit as st
import pandas as pd
import altair as alt
from recommender import Recommender
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from os import cpu_count
import numpy as np
import time

from utils import load_and_preprocess_data

import matplotlib.pyplot as plt
from typing import Union, List, Dict, Any
import plotly.graph_objects as go


COLUMN_NOT_DISPLAY = [
    "StockCode",
    "UnitPrice",
    "Country",
    "CustomerIndex",
    "ProductIndex",
]


SIDEBAR_DESCRIPTION = """
# Recommender system

## What is it?
A recommender system is a tool that suggests something new to a particular
user that she/he might be interested in. It becomes useful when
the number of items a user can choose from is high.

## How does it work?
A recommender system internally finds similar users and similar items,
based on a suitable definition of "similarity".
For example, users that purchased the same items can be considered similar.
When we want to suggest new items to a user, a recommender system exploits
the items bought by similar users as a starting point for the suggestion. 
The items bought by similar users are compared to the items that the user
already bought. If they are new and similar, the model suggests them.

## How we prepare the data
For each user, we compute the quantity purchased for every single item. 
This will be the metric the value considered by the model to compute 
the similarity. The item that a user has never bought will
be left at zero. These zeros will be the subject of the recommendation.
""".lstrip()


@st.cache(allow_output_mutation=True)
def create_and_fit_recommender(
    model_name: str,
    values: Union[pd.DataFrame, "np.ndarray"],
    users: Union[pd.DataFrame, "np.ndarray"],
    products: Union[pd.DataFrame, "np.ndarray"],
) -> Recommender:
    recommender = Recommender(
        values,
        users,
        products,
    )

    recommender.create_and_fit(
        model_name,
        # Fine-tuned values
        model_params=dict(
            factors=190,
            alpha=0.6,
            regularization=0.06,
            random_state=42,
        ),
    )
    return recommender


def explain_recommendation(
    recommender: Recommender,
    user_id: int,
    suggestions: List[int],
    df: pd.DataFrame,
):
    output = []

    n_recommended = len(suggestions)
    for suggestion in suggestions:
        explained = recommender.explain_recommendation(
            user_id, suggestion, n_recommended
        )

        suggested_items_id = [id[0] for id in explained]

        suggested_description = (
            df.loc[df.ProductIndex == suggestion][["Description", "ProductIndex"]]
            .drop_duplicates(subset=["ProductIndex"])["Description"]
            .unique()[0]
        )
        similar_items_description = (
            df.loc[df["ProductIndex"].isin(suggested_items_id)][
                ["Description", "ProductIndex"]
            ]
            .drop_duplicates(subset=["ProductIndex"])["Description"]
            .unique()
        )

        output.append(
            f"The item **{suggested_description.strip()}** "
            "has been suggested because it is similar to the following products"
            " bought by the user:"
        )
        for description in similar_items_description:
            output.append(f"- {description.strip()}")

    with st.expander("See why the model recommended these products"):
        st.write("\n".join(output))

    st.write("------")


def print_suggestions(suggestions: List[int], df: pd.DataFrame):
    similar_items_description = (
        df.loc[df["ProductIndex"].isin(suggestions)][["Description", "ProductIndex"]]
        .drop_duplicates(subset=["ProductIndex"])["Description"]
        .unique()
    )

    output = ["The model suggests the following products:"]
    for description in similar_items_description:
        output.append(f"- {description.strip()}")

    st.write("\n".join(output))


def display_user_char(user: int, data: pd.DataFrame):
    subset = data[data.CustomerIndex == user]
    # products = subset.groupby("ProductIndex").agg(
    #     {"Description": lambda x: x.iloc[0], "Quantity": sum}
    # )

    st.write(
        "The user {} bought {} distinct products. Here is the purchase history: ".format(
            user, subset["Description"].nunique()
        )
    )
    st.dataframe(
        subset.sort_values("InvoiceDate").drop(
            # Do not show the customer since we are display the
            # information for a specific customer.
            COLUMN_NOT_DISPLAY + ["CustomerID"],
            axis=1,
        )
    )
    st.write("-----")


def _extract_description(df, products):
    desc = df[df["ProductIndex"].isin(products)].drop_duplicates(
        "ProductIndex", ignore_index=True
    )[["ProductIndex", "Description"]]
    return desc.set_index("ProductIndex")


def display_recommendation_plots(
    user_id: int,
    suggestions: List[int],
    df: pd.DataFrame,
    model: Recommender,
):
    """Plots a t-SNE with the suggested items, togheter with the purchases of
    similar users.
    """
    # Get the purchased items that contribute the most to the suggestions
    contributions = []
    n_recommended = len(suggestions)
    for suggestion in suggestions:
        items_and_score = model.explain_recommendation(
            user_id, suggestion, n_recommended
        )
        contributions.append([t[0] for t in items_and_score])

    contributions = np.unique(np.concatenate(contributions))

    print("Contribution computed")
    print(contributions)
    print("=" * 80)

    # Find the purchases of similar users
    bought_by_similar_users = []

    sim_users, _ = model.similar_users(user_id)

    for u in sim_users:
        _, sim_purchases = model.user_product_matrix[u].nonzero()
        bought_by_similar_users.append(sim_purchases)

    bought_by_similar_users = np.unique(np.concatenate(bought_by_similar_users))

    print("Similar bought computed")
    print(bought_by_similar_users)
    print("=" * 80)

    # Compute the t-sne

    # Concate all the vectors to compute a single time the decomposition
    to_decompose = np.concatenate(
        (
            model.item_factors[suggestions],
            model.item_factors[contributions],
            model.item_factors[bought_by_similar_users],
        )
    )

    print(f"Shape to decompose: {to_decompose.shape}")

    with st.spinner("Computing plots (this might take around 60 seconds)..."):
        elapsed = time.time()
        decomposed = _tsne_decomposition(
            to_decompose,
            dict(
                perplexity=30,
                metric="euclidean",
                n_iter=1_000,
                random_state=42,
            ),
        )
    elapsed = time.time() - elapsed
    print(f"TSNE computed in {elapsed}")
    print("=" * 80)

    # Extract the decomposed vectors
    suggestion_dec = decomposed[: len(suggestions), :]
    contribution_dec = decomposed[
        len(suggestions) : len(suggestions) + len(contributions), :
    ]
    items_others_dec = decomposed[-len(bought_by_similar_users) :, :]

    # Also, extract the description to create a nice hover in
    # the final plot.

    contribution_description = _extract_description(df, contributions)
    items_other_description = _extract_description(df, bought_by_similar_users)
    suggestion_description = _extract_description(df, suggestions)

    # Plot the scatterplot

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=contribution_dec[:, 0],
            y=contribution_dec[:, 1],
            mode="markers",
            opacity=0.8,
            name="Similar bought by user",
            marker_symbol="square-open",
            marker_color="#010CFA",
            marker_size=10,
            hovertext=contribution_description.loc[contributions].values.squeeze(),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=items_others_dec[:, 0],
            y=items_others_dec[:, 1],
            mode="markers",
            name="Product bought by similar users",
            opacity=0.7,
            marker_symbol="circle-open",
            marker_color="#FA5F19",
            marker_size=10,
            hovertext=items_other_description.loc[
                bought_by_similar_users
            ].values.squeeze(),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=suggestion_dec[:, 0],
            y=suggestion_dec[:, 1],
            mode="markers",
            name="Suggested",
            marker_color="#1A9626",
            marker_symbol="star",
            marker_size=10,
            hovertext=suggestion_description.loc[suggestions].values.squeeze(),
        )
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(plot_bgcolor="white")

    return fig


def _tsne_decomposition(data: np.ndarray, tsne_args: Dict[str, Any]):
    if data.shape[1] > 50:
        print("Performing PCA...")
        data = PCA(n_components=50).fit_transform(data)
    return TSNE(
        n_components=2,
        n_jobs=cpu_count(),
        **tsne_args,
    ).fit_transform(data)


def main():
    # Load and process data
    data, users, products = load_and_preprocess_data()
    recommender = create_and_fit_recommender(
        "als",
        data["Quantity"],
        users,
        products,
    )

    st.markdown(
        """# Recommender system
The dataset used for these computations is the following:
        """
    )
    st.sidebar.markdown(SIDEBAR_DESCRIPTION)

    to_display = data.drop(
        COLUMN_NOT_DISPLAY,
        axis=1,
    )

    # Convert to int just to display the column without trailing decimals.
    # @note: I know I can use the "format" function of pandas, but I found out
    #   it is super slow when fomratting large tables.
    to_display["Price"] = to_display["Price"].astype(int)

    # Show the data
    st.dataframe(
        to_display,
    )

    st.markdown("## Interactive suggestion")
    with st.form("recommend"):
        # Let the user select the user to investigate
        user = st.selectbox(
            "Select a customer to get his recommendations",
            users.unique(),
        )

        items_to_recommend = st.slider("How many items to recommend?", 1, 10, 5)
        print(items_to_recommend)

        submitted = st.form_submit_button("Recommend!")
        if submitted:
            # show_purhcase_history(user, data)
            display_user_char(user, data)
            suggestions_and_score = recommender.recommend_products(
                user, items_to_recommend
            )
            print_suggestions(suggestions_and_score[0], data)
            explain_recommendation(recommender, user, suggestions_and_score[0], data)

            st.markdown(
                "## How the purchases of similar users influnce the recommendation"
            )
            fig = display_recommendation_plots(
                user, suggestions_and_score[0], data, recommender
            )
            st.plotly_chart(fig)


main()