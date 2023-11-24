---
title: Recommender system and customer segmentation
emoji: 🐨
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.10.0
app_file: recommender_system.py
pinned: false
license: mit
---

# Recommender system and customer segmentation

Demo with recsys and clustering for the [online retail](https://www.kaggle.com/datasets/vijayuv/onlineretail?select=OnlineRetail.csv) dataset.

## Objective

Recommender system:

    1. interactively select a user
    2. show all the recommendations for the user
    3. explain why we get these suggestions (which purchased object influences the most)
    4. plot the purchases and suggested articles 
    
Clustering:
    
    1. compute the user clustering
    2. plot users and their clusters
    3. explain the meaning of the clusters (compute the mean metrics or literally explain them)
## Setup

In your terminal run:

```bash
# Enable the env
source .venv/bin/activate
# Install the dependencies
pip install -r requirements.txt
# Or install the freezed dependencies from the requirements_freezed.txt
# You are ready to rock!
```

## Run

In your terminal run:

```bash
streamlit run recommender_system.py
# Now the defualt browser will be opened with 
# the stramlit page. It you want to customize the
# execution of streaming, refer to its documentation.
```

## Resources

- [streamlit](https://streamlit.io/)
- [implicit](https://github.com/benfred/implicit), recsys library
- [t-sne guide](https://distill.pub/2016/misread-tsne/)
- [RFM segmentation](https://www.omniconvert.com/blog/rfm-score/)

# ----------------------------
# 2) Examples of Collaborative Filtering Recommender Systems:
#
##  a. Netflix
##  b. Amazon
##  c. Spotify
##  d. Linkedin
#
# 1) Examples of Content-Based Recommender Systems:
#
##  a. YouTube
##  b. Pandora
##  c. Pinterest
##  d. Medium
#
# ----------------------------





