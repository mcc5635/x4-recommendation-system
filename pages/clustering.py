from collections import defaultdict
import streamlit as st
from utils import load_and_preprocess_data
import pandas as pd
import numpy as np
import altair as alt
from sklearn.mixture import GaussianMixture
import plotly.express as px
import itertools
from typing import Dict, List, Tuple


SIDEBAR_DESCRIPTION = """
# Client clustering

To cluster a client, we adopt the RFM metrics. They stand for:

- R = recency, that is the number of days since the last purchase
    in the store
- F = frequency, that is the number of times a customer has ordered something
- M = monetary value, that is how much a customer has spent buying
    from your business.

Given these 3 metrics, we can cluster the customers and find a suitable
"definition" based on the clusters they belong to. Since the dataset
we're using right now has about 5000 distinct customers, we identify
3 clusters for each metric.

## How we compute the clusters

We resort to a GaussianMixture algorithm. We can think of GaussianMixture
as generalized k-means clustering that incorporates information about
the covariance structure of the data as well as the centers of the clusters.
""".lstrip()

FREQUENCY_CLUSTERS_EXPLAIN = """
The **frequency** denotes how frequently a customer has ordered.

There 3 available clusters for this metric:

- cluster 1: denotes a customer that purchases one or few times (range [{}, {}])
- cluster 2: these customer have a discrete amount of orders (range [{}, {}])
- cluster 3: these customer purchases lots of times (range [{}, {}])

-------
""".lstrip()

RECENCY_CLUSTERS_EXPLAIN = """
The **recency** refers to how recently a customer has bought;

There 3 available clusters for this metric:

- cluster 1: the last order of these client is long time ago (range [{}, {}])
- cluster 2: these are clients that purchases something not very recently (range [{}, {}])
- cluster 3: the last order of these client is a few days/weeks ago (range [{}, {}])

-------
""".lstrip()

MONETARY_CLUSTERS_EXPLAIN = """
The **revenue** refers to how much a customer has spent buying
from your business.

There 3 available clusters for this metric:

- cluster 1: these clients spent little money (range [{}, {}])
- cluster 2: these clients spent a considerable amount of money (range [{}, {}])
- cluster 3: these clients spent lots of money (range [{}, {}])

-------
""".lstrip()

EXPLANATION_DICT = {
    "Frequency_cluster": FREQUENCY_CLUSTERS_EXPLAIN,
    "Recency_cluster": RECENCY_CLUSTERS_EXPLAIN,
    "Revenue_cluster": MONETARY_CLUSTERS_EXPLAIN,
}


def create_features(df: pd.DataFrame):
    """Creates a new dataframe with the RFM features for each client."""
    # Compute frequency, the number of distinct time a user purchased.
    client_features = df.groupby("CustomerID")["InvoiceDate"].nunique().reset_index()
    client_features.columns = ["CustomerID", "Frequency"]

    # Add monetary value, the total revenue for  each single user.
    client_takings = df.groupby("CustomerID")["Price"].sum()
    client_features["Revenue"] = client_takings.values

    # Add recency, i.e. the days since the last purchase in the store.
    max_date = df.groupby("CustomerID")["InvoiceDate"].max().reset_index()
    max_date.columns = ["CustomerID", "LastPurchaseDate"]

    client_features["Recency"] = (
        max_date["LastPurchaseDate"].max() - max_date["LastPurchaseDate"]
    ).dt.days

    return client_features


@st.cache
def cluster_clients(df: pd.DataFrame):
    """Computes the RFM features and clusters for each user based on the RFM metrics."""

    df_rfm = create_features(df)

    for to_cluster, order in zip(
        ["Revenue", "Frequency", "Recency"], ["ascending", "ascending", "descending"]
    ):
        kmeans = GaussianMixture(n_components=3, random_state=42)
        labels = kmeans.fit_predict(df_rfm[[to_cluster]])
        df_rfm[f"{to_cluster}_cluster"] = _order_cluster(kmeans, labels, order)

    return df_rfm


def _order_cluster(cluster_model: GaussianMixture, clusters, order="ascending"):
    """Orders the cluster by `order`."""
    centroids = cluster_model.means_.sum(axis=1)

    if order.lower() == "descending":
        centroids *= -1

    ascending_order = np.argsort(centroids)
    lookup_table = np.zeros_like(ascending_order)
    # Cluster will start from 1
    lookup_table[ascending_order] = np.arange(cluster_model.n_components) + 1
    return lookup_table[clusters]


def show_purhcase_history(user: int, df: pd.DataFrame):
    user_purchases = df.loc[df.CustomerID == user, ["Price", "InvoiceDate"]]
    expenses = user_purchases.groupby(user_purchases.InvoiceDate).sum()
    expenses.columns = ["Expenses"]
    expenses = expenses.reset_index()

    c = (
        alt.Chart(expenses)
        .mark_line(point=True)
        .encode(
            x=alt.X("InvoiceDate", timeUnit="yearmonthdate", title="Date"),
            y="Expenses",
        )
        .properties(title="User expenses")
    )

    st.altair_chart(c)


def show_user_info(user: int, df_rfm: pd.DataFrame):
    """Prints some information about the user.

    The main information are the total expenses, how
    many times he purchases in the store, and the clusters
    he belongs to.
    """

    user_row = df_rfm[df_rfm["CustomerID"] == user]
    if len(user_row) == 0:
        st.write(f"No user with id {user}")

    output = []

    output.append(f"The user purchased **{user_row['Frequency'].squeeze()} times**.\n")
    output.append(
        f"She/he spent **{user_row['Revenue'].squeeze()} dollars** in total.\n"
    )
    output.append(
        f"The last time she/he bought something was **{user_row['Recency'].squeeze()} days ago**.\n"
    )
    output.append(f"She/he belongs to the clusters: ")
    for cluster in [column for column in user_row.columns if "_cluster" in column]:
        output.append(f"- {cluster} = {user_row[cluster].squeeze()}")

    st.write("\n".join(output))

    return (
        user_row["Recency_cluster"].squeeze(),
        user_row["Frequency_cluster"].squeeze(),
        user_row["Revenue_cluster"].squeeze(),
    )


def explain_cluster(cluster_info):
    """Displays a popup menu explinging the meanining of the clusters."""

    with st.expander("Show information about the clusters"):
        st.write(
            "**Note**: these values are valid for these dataset."
            "Different dataset will have different number of clusters"
            " and values"
        )
        for cluster, info in cluster_info.items():
            # Transform the (mins, maxs) tuple into
            # [min_1, max_1, min_2, max_2, ...] list.
            min_max_interleaved = list(itertools.chain(*zip(info[0], info[1])))
            st.write(EXPLANATION_DICT[cluster].format(*min_max_interleaved))


def categorize_user(recency_cluster, frequency_cluster, monetary_cluster):
    """Describe the user with few words based on the cluster he belongs to."""

    score = f"{recency_cluster}{frequency_cluster}{monetary_cluster}"

    # @fixme: find a better approeach. These elif chains don't scale at all.

    description = ""

    if score == "111":
        description = "Tourist"
    elif score.startswith("2"):
        description = "Losing interest"
    elif score == "133":
        description = "Former lover"
    elif score == "123":
        description = "Former passionate client"
    elif score == "113":
        description = "Spent a lot, but never come back"
    elif score.startswith("1"):
        description = "About to dump"
    elif score == "313":
        description = "Potential lover"
    elif score == "312":
        description = "Interesting new client"
    elif score == "311":
        description = "New customer"
    elif score == "333":
        description = "Gold client"
    elif score == "322":
        description = "Lovers"
    else:
        description = "Average client"

    st.write(f"The customer can be described as: **{description}**")


def plot_rfm_distribution(
    df_rfm: pd.DataFrame, cluster_info: Dict[str, Tuple[List[int], List[int]]]
):
    """Plots 3 histograms for the RFM metrics."""

    for x, to_reverse in zip(("Revenue", "Frequency", "Recency"), (False, False, True)):
        fig = px.histogram(
            df_rfm,
            x=x,
            log_y=True,
            title=f"{x} metric",
        )
        # Get the max value in the cluster info. The cluster_info_dict is a
        # tuple with first element the min values of the cluster, and second
        # element the max values of the cluster.
        values = cluster_info[f"{x}_cluster"][1]  # get max values
        print(values)
        # Add vertical bar on each cluster end. But skip the last cluster.
        loop_range = range(len(values) - 1)
        if to_reverse:
            # Skip the last element
            loop_range = range(len(values) - 1, 0, -1)
        for n_cluster in loop_range:
            print(x)
            print(values[n_cluster])
            fig.add_vline(
                x=values[n_cluster],
                annotation_text=f"End of cluster {n_cluster+1}",
                line_dash="dot",
                annotation=dict(textangle=90, font_color="red"),
            )

        fig.update_layout(
            yaxis_title="Count (log scale)",
        )

        st.plotly_chart(fig)


def display_dataframe_heatmap(df_rfm: pd.DataFrame, cluster_info_dict):
    """Displays an heatmap of how many clients lay in the clusters.

    This method uses some black magic coming from the dataframe
    styling guide.
    """

    def style_with_limits(x, column, cluster_limit_dict):
        """Simple function to transform the cluster number into
        a cluster + range string."""
        min_v = cluster_limit_dict[column][0][x - 1]
        max_v = cluster_limit_dict[column][1][x - 1]
        return f"{x}: [{int(min_v)}, {int(max_v)}]"

    # Create a dataframe with the count of clients for each group
    # of cluster.

    count = (
        df_rfm.groupby(["Recency_cluster", "Frequency_cluster", "Revenue_cluster"])[
            "CustomerID"
        ]
        .count()
        .reset_index()
    )
    count = count.rename(columns={"CustomerID": "Count"})

    # Remove duplicates
    count = count.drop_duplicates(
        ["Revenue_cluster", "Frequency_cluster", "Recency_cluster"]
    )

    # Add limits to the cells. In this way, we can better display
    # the heatmap.
    for cluster in ["Revenue_cluster", "Frequency_cluster", "Recency_cluster"]:
        count[cluster] = count[cluster].apply(
            lambda x: style_with_limits(x, cluster, cluster_info_dict)
        )

    # Use the count column as values, then index with the clusters.
    count = count.pivot(
        index=["Revenue_cluster", "Frequency_cluster"],
        columns="Recency_cluster",
        values="Count",
    )

    # Style manipulation
    cell_hover = {
        "selector": "td",
        "props": "font-size:1.2em",
    }
    index_names = {
        "selector": ".index_name",
        "props": "font-style: italic; color: Black; font-weight:normal;font-size:1.2em;",
    }
    headers = {
        "selector": "th:not(.index_name)",
        "props": "background-color: White; color: black; font-size:1.2em",
    }

    # Finally, display
    # We cannot directly print the dataframe since the streamlit
    # functin remove the multiindex. Thus, we extract the html representation
    # and then display it.
    st.markdown("## Heatmap: how the client are distributed between clusters")
    st.write(
        count.style.format(thousands=" ", precision=0, na_rep="0")
        .set_table_styles([cell_hover, index_names, headers])
        .background_gradient(cmap="coolwarm")
        .to_html(),
        unsafe_allow_html=True,
    )


def main():
    st.sidebar.markdown(SIDEBAR_DESCRIPTION)

    df, _, _ = load_and_preprocess_data()
    df_rfm = cluster_clients(df)

    st.markdown(
        "# Dataset "
        "\nThis is the processed dataset with information about the clients, such as"
        " the RFM values and the clusters they belong to."
    )
    st.dataframe(df_rfm.style.format(formatter={"Revenue": "{:.2f}"}))

    cluster_info_dict = defaultdict(list)

    with st.expander("Show more details about the clusters"):
        for cluster in [column for column in df_rfm.columns if "_cluster" in column]:
            st.write(cluster)
            cluster_info = (
                df_rfm.groupby(cluster)[cluster.split("_")[0]]
                .describe()
                .reset_index(names="Cluster")
            )
            min_cluster = cluster_info["min"].astype(int)
            max_cluster = cluster_info["max"].astype(int)
            cluster_info_dict[cluster] = (min_cluster, max_cluster)
            st.dataframe(cluster_info)

    st.markdown("## RFM metric distribution")

    plot_rfm_distribution(df_rfm, cluster_info_dict)

    display_dataframe_heatmap(df_rfm, cluster_info_dict)

    st.markdown("## Interactive exploration")

    filter_by_cluster = st.checkbox(
        "Filter client: only one client per cluster type",
        value=True,
    )

    client_to_select = (
        df_rfm.groupby(["Recency_cluster", "Frequency_cluster", "Revenue_cluster"])[
            "CustomerID"
        ]
        .first()
        .values
        if filter_by_cluster
        else df["CustomerID"].unique()
    )

    # Let the user select the user to investigate
    user = st.selectbox(
        "Select a customer to show more information about him.",
        client_to_select,
    )

    show_purhcase_history(user, df)

    recency, frequency, revenue = show_user_info(user, df_rfm)

    categorize_user(recency, frequency, revenue)

    explain_cluster(cluster_info_dict)


main()