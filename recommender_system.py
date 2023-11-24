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
    "ProductIndex"
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





