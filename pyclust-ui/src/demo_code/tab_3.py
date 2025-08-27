import json
import os

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyclustkit.metalearning import  MFExtractor


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
from sklearn.model_selection import LeaveOneOut

from collections import  Counter

import pickle

from demo_code.generic import meta_learners_repository


mfe = MFExtractor()
mf_categories = mfe.mf_categories
mf_papers = mfe.mf_papers
all_mf = [key for key in mfe.meta_features]



mfe.search_mf(category="descriptive", search_type="names")


def update_ml_options(selected_ml):
    """
    Visibility of meta-lerner classifier parameters
    Args:
        selected_ml (str):

    Returns:

    """
    if selected_ml == "KNN":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
    elif selected_ml == "DT":
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)


def update_search_type_dropdown(method):
    var = {"Category": mf_categories,
           "Paper": mf_papers,
           "Custom Selection": all_mf}
    return gr.update(value="", choices=var[method])


def toggle_mf_selection_options(method):
    if method == "Custom Selection":
        return gr.update(visible=True), gr.update(visible=True)
    elif method == "All":
        return gr.update(visible=False), gr.update(visible=False)

def train_meta_learner(algorithm, mf, best_alg, *alg_options ):
    # ---(1)--- Configure Classification algorithm
    if algorithm == "KNN":
        alg = KNeighborsClassifier(n_neighbors=alg_options[0], metric=alg_options[1])
    elif algorithm == "DT":
        alg = DecisionTreeClassifier(criterion=alg_options[2])

    # ---(2)--- Filter selected Meta-features and create dataframe
    mfdf = pd.DataFrame()

    if mf == "Custom Selection" :
        mf_to_include = []

        if alg_options[3] == "Category" :
            for cat in alg_options[4]:
                mf_to_include += mfe.search_mf(category=cat, search_type="names")

        elif alg_options[3] == "Paper" :
            for paper in alg_options[4]:
                mf_to_include += mfe.search_mf(included_in=paper, search_type="names")

    elif mf == "All":
        mf_to_include = mfe.search_mf(search_type="names")

    # ---> Parse <repository> folder for json files that contain meta-features
    base_path = os.path.join(os.getcwd(), "repository/meta_features")
    for dirpath, dirnames, filenames in os.walk(base_path):
        for filename in filenames:
            if filename.endswith(".json"):
                file_path = os.path.join(dirpath, filename)
                with open(file_path, "r") as f:
                    mf_dict = json.load(f)
                    mf_dict = {k: v for k, v in mf_dict.items() if k in mf_to_include}

                    mf_df = pd.json_normalize(mf_dict)
                    mf_df['dataset'] = filename.replace(".json", "").replace("_mf", "").replace("_", "-")

                    mfdf = pd.concat([mfdf, mf_df], ignore_index=True)

    mfdf = mfdf.reset_index(drop=True)
    print(f"Meta Features loaded successfully with {mfdf.shape}")

    # ---(3)--- Get Labels
    input_dir = "repository/best_alg_per_cvi"
    master_cvi_dict = {}

    for dirpath, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith(".json"):
                file_path = os.path.join(dirpath, filename)
                with open(file_path, "r") as f:
                    key = filename.replace(".json", "").replace("_best_alg","")
                    master_cvi_dict[key] = json.load(f)

    best_alg_per_dataset = []
    if best_alg == "Most Popular Alg":
        for key in master_cvi_dict:
            count_cvi_pop = Counter(list(master_cvi_dict[key].values()))
            most_pop = count_cvi_pop.most_common(1)[0][0]
            best_alg_per_dataset.append(( key.replace(".json", "").replace("_", "-"), most_pop))

    elif best_alg == "Specific CVI":
        cvi_selected = alg_options[5]
        for key in master_cvi_dict:
            best_alg_per_dataset.append(( key.replace(".json", "").replace("_", "-"), master_cvi_dict[key][cvi_selected]))

    labels_df = pd.DataFrame(best_alg_per_dataset)
    labels_df.columns = ["dataset", "algorithm"]


    master_df = pd.merge(mfdf, labels_df, on="dataset", how="outer").reset_index(drop=True)
    master_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    master_df = master_df.fillna(0)


    # ---(4)--- Algorithm Training
    X = master_df.drop(columns=["dataset", "algorithm"])
    y = master_df["algorithm"]
    print("-0--------X")
    print(X)
    print("-0--------y")
    print(y)
    y_true, y_pred = [], []
    # If the datasets are few, evaluate with Leave-One-Out
    if master_df.shape[0] <= 40:
        loo = LeaveOneOut()

        for train_index, test_index in loo.split(master_df):
            x_train, x_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            alg.fit(x_train, y_train)

            y_true.append(y_test.iloc[0])  # Single test instance
            y_pred.append(alg.predict(x_test.iloc[0].values.reshape(1,-1))[0])
    else:
        loo = LeaveOneOut()

        for train_index, test_index in loo.split(master_df):
            x_train, x_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            alg.fit(x_train, y_train)

            y_true.append(y_test.iloc[0])  # Single test instance
            y_pred.append(alg.predict(x_test.iloc[0].values.reshape(1, -1))[0])

    # ---(5)--- Return Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    label_map = {"AgglomerativeClustering" : "Agglomerative", "AffinityPropagation": "Af.Propagation"}
    labels = [label_map[label] if label in label_map.keys() else label for label in np.unique(y) ]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues)

    plt.xticks(rotation=45, fontsize=10, ha='right')  # Rotate x-axis labels
    plt.yticks(fontsize=7)

    plt.tight_layout()
    plt.savefig("confusion_matrix_loocv.png")
    plt.close()

    gr.Info("Meta Learner trained successfully!")
    ml_meta_data  = {"no_datasets": master_df.shape[0],
                     "evaluation": {"method": "Leave-One-Out", "accuracy": accuracy_score(y_true, y_pred)},
                     "meta-features": {"number": len(mfdf.columns), "based_on": alg_options[3], "selection": alg_options[4]},
                     "best_algorithm": {"based_on": best_alg, "selection": alg_options[5]},
                     "classes_that_appear_in_data": len(np.unique(y)) }

    return "confusion_matrix_loocv.png", ml_meta_data, alg


def save_meta_learner(model, model_name, model_metadata):
    """
    Saves the trained meta-learner and updates the metadata file.

    Args:
        model_metadata (dict): The metadata object of the meta-learner
        model (sklearn.base.BaseEstimator): Any meta-learner classification model trained (from scikit learn)
        model_name (str): The id the user provides

    Returns:

    """
    # ---(1)--- Check if model name is empty. If so, it's replaced with _ to avoid internal errors
    if model_name == "":
        model_name = "_"

    # ---(2)--- Check if any meta-learning model with the same name exists
    meta_learners_saved = os.listdir("repository/meta_learners/models")
    if model_name + ".pkl" in meta_learners_saved:
        raise gr.Error("A meta-learner with the same ID is present in the repository.")

    # ---(3)--- Save the trained meta-learning model
    with open(f"repository/meta_learners/models/{model_name}.pkl", "wb") as f:
        pickle.dump(model, f)
    gr.Info("Model saved Successfully!")

    # ---(4)--- Update meta-learner metadata in the repository
    with open(f"repository/meta_learners/meta-data.json", "r") as f:
        ml_metadata = json.load(f)

    ml_metadata.update({model_name:model_metadata})

    with open(f"repository/meta_learners/meta-data.json", "w") as f:
        json.dump(ml_metadata, f)

    gr.Info("Meta-Learner Metadata Repository Updated Successfully!")

    mldf, ml_choices = meta_learners_repository()

    return mldf, gr.update(choices=ml_choices)