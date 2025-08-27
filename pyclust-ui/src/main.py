import gradio.themes

from demo_code.generic import *
from demo_code.tab_1 import *
from demo_code.tab_2 import *
from demo_code.tab_3 import *

import os

# On demo startup: (1) load trained meta-learner metadata, (2) Load number of datasets
mldf, ml_choices = meta_learners_repository()
no_datasets = on_startup_read_no_datasets()


with gr.Blocks(theme=gradio.themes.Default(text_size="lg"), css_paths=os.path.join("demo_code", "demo_style.css")) as demo:
    # Cache data
    df = gr.State()
    data_id = gr.State()

    df_needs_reduction = gr.State()
    df_reduced = gr.State({"MDS": None, "PCA": None, "T-SNE": None})

    operations_complete = gr.State({"meta-features-extraction": False, "configurations-search": False,
                                    "results_saved_to_repo": False})

    meta_learner_trained = gr.State()
    trained_meta_learner_metadata = gr.State()

    best_config_per_cvi = gr.State({})

    # <-----------------------------Demo Titles ✓-------------------------------------------------------------->

    demo_title = gr.Markdown("<h1 style='text-align: center; overflow: hidden; font-size: 50px;'>PyClust Demo</h1>")

    demo_title_dataset_loaded = gr.Markdown("<h2 style='text-align: center; color:red;'>No dataset loaded!</h2>")
    with gr.Row(visible=False) as title_row:
        demo_mf_completed = gr.Markdown("<h2 style='text-align: right; color:#3ebefe;'>Meta-Features: ❌</h2>")
        demo_ms_completed = gr.Markdown("<h2 style='text-align: left; color:#3ebefe;'>Model Search: ❌</h2>")

    with gr.Row(visible=False) as title_row_2:
        with gr.Column(0.9):
            add_data_to_repo_title = gr.Markdown(
                "<h2 style='text-align: right; color:#3ebefe;'>Dataset in Repository: ❌</h2>")
        with gr.Column(scale=0.8):
            add_data_to_repo_btn = gr.Button("Add", size="sm", elem_id="title_button")

    # <-----------------------------Data Loading ✓-------------------------------------------
    with gr.Tab('(1) Data Loading/Online Phase'):
        with gr.Accordion("Usage Manual", open=True, elem_id="my_accordion"):
            gr.Markdown("""
                        # Welcome to the **PyClust Demo**! 🎉

                        ## Get Started:
                        On this page, you can:
                        1. **Load or Generate Data** 📊  
                        2. **Compute Meta-Features** 🔍  
                        3. **Select the Best Algorithm** for your dataset 🧠  
                        
                        ## What's Next?
                        - Dive into **Model Search** to train and evaluate different models.  
                        - Explore the **Repository** for pre-trained model selection meta-learners or train  and train new.. 

                        """)

        with gr.Accordion("Generate/Load Data", open=False, elem_id="my_accordion"):
            # Only visible after dataset is loaded or generated.
            with gr.Column(visible=False) as data_success_row:
                success_msg = gr.Markdown()
                change_data_btn = gr.Button("Change Dataset")

            with gr.Row() as data_method_row:
                data_id_textbox = gr.Textbox(label="Assign dataset ID", elem_classes="center-label")
                data_method = gr.Radio(choices=["Upload", "Generate"], label="Select How to Provide Data",
                                       elem_classes="center-radio")

                # - Data Upload Column
            with gr.Column(visible=False) as data_upload_row:
                gr.Markdown(
                    "Warning!: PyClust has been developed for numeric data. Please only upload datasets in .csv form "
                    "with scalar values.")
                upload_csv_options = gr.CheckboxGroup(["Headers", "Scale Data (MinMax)"],
                                                      label="Preprocessing Options")
                csv_file = gr.File(label="Upload your CSV file")

                # - Data Generation Column

            with gr.Column(visible=False) as data_generation_row:
                synthetic_data_method = gr.Radio(choices=["Blobs", "Moons"],
                                                 label="Choose Synthetic Data Generation Type", value="Blobs")
                no_instances_input = gr.Number(label='No Instances', value=100)
                no_features_input = gr.Number(label='No Features', value=2)
                generate_data_btn = gr.Button('Generate Synthetic Data')

        with gr.Accordion("Calculate Meta Features", open=False, elem_id="my_accordion"):
            mf_calculate_btn = gr.Button(value="Calculate MF")
            download_mf_btn = gr.DownloadButton("Download (JSON)", visible=False)
            mf_calculated = gr.JSON()

        # <-----------------------------Data Loading - Prediction------------------------------------->
        with gr.Accordion("Algorithm Selection!", open=False, elem_id="my_accordion"):
            prediction = gr.Markdown("")
            with gr.Column():
                ml_model_choice = gr.Dropdown(label="select Meta Learner", choices=ml_choices, interactive=True)
                predict_btn = gr.Button("Predict !")

        # On change/Click for tab 1
        csv_file.change(load_csv, inputs=[csv_file, upload_csv_options, data_id_textbox],
                        outputs=[data_method_row, data_upload_row, data_generation_row, data_id_textbox,
                                 data_success_row, success_msg, df, data_id, demo_title_dataset_loaded, title_row])

        generate_data_btn.click(generate_data, inputs=[synthetic_data_method, no_instances_input,
                                                       no_features_input, data_id_textbox],
                                outputs=[data_method_row, data_upload_row, data_generation_row,
                                         data_success_row, success_msg, df, data_id,
                                         demo_title_dataset_loaded, title_row])

        change_data_btn.click(change_data_update_visibility, inputs=operations_complete,
                              outputs=[data_method_row, data_success_row, data_method, operations_complete])

        data_method.change(control_data_visibility, inputs=data_method, outputs=[data_generation_row,
                                                                                 data_upload_row])

        mf_calculate_btn.click(mf_process, inputs=[df, operations_complete], outputs=[mf_calculated, download_mf_btn,
                                                                          operations_complete])

        predict_btn.click(load_model_and_predict, inputs=[ml_model_choice, data_id ],
                          outputs=[prediction])

    # <-----------------------------Exhaustive Search ✓-------------------------------------------------------------->
    with gr.Tab('(2) Parameter Search'):
        not_loaded_message = gr.Markdown("""
                                        <div style="display: flex; justify-content: center; align-items: center; 
                                        height: 100%;">
                                            No data loaded yet.
                                        </div>
                                        """, elem_id="centered-message")

        with gr.Tab("Grid Search", visible=False) as model_search_tab:
            with gr.Column() as es_col:
                with gr.Column():
                    es_start_btn = gr.Button("Start Exhaustive Search")
                    search_space_input = gr.Textbox(value=search_space, label='Set Search Space',
                                                    interactive=True,
                                                    lines=12)

            with gr.Column() as es_success_row:
                es_success_msg = gr.Markdown(visible=False)
                dl_results_btn = gr.DownloadButton("Download Results", "es_search_results.json", visible=False)

            with gr.Column() as es_results_row:
                gr.Markdown("<h1 style='text-align: center;'>Exhaustive Search Results</h1>")
                with gr.Row():
                    hist_img = gr.Image(interactive=False)
                    pie_img = gr.Image(interactive=False)

            es_start_btn.click(exhaustive_search, inputs=[ data_id, df, search_space_input,
                                                          operations_complete],
                               outputs=[ es_success_msg, dl_results_btn, pie_img, hist_img,
                                        best_config_per_cvi, operations_complete])

        # <-----------------------------Clustering Exploration ✓--------------------------------------------------------->
        with gr.Tab('Clustering Exploration', visible=False) as clustering_exploration_tab:
            gr.Markdown(""" In this page you can search the exhaustive search results for the best model as indicated 
                                        by a cvi and visually explore the clusters. """)
            best_config_row = gr.Row(visible=True, equal_height=True)
            reduction_column = gr.Column(visible=False)
            results_explore_column = gr.Column(visible=True)

            with best_config_row:
                with gr.Column(scale=3):
                    best_config_index_dropdown = gr.Dropdown(choices=cvi_list, multiselect=False,
                                                             label="Select Index", visible=True, interactive=True,
                                                             elem_id="my-dropdown", value="")
                with gr.Column(scale=7):
                    best_config = gr.Textbox(label="Best Configuration", elem_id="my-textbox")

            with reduction_column:
                gr.Markdown("""Provided Dataset has more than two Features. 
                            Please select a method to reduce dimensions to 2.""")

                reduction_choices = gr.Radio(choices=["PCA", "T-SNE", "MDS"], value="PCA")

            with results_explore_column:
                with gr.Row():
                    with gr.Column():
                        # Dim Reduction Functionality.

                        # reduction_choices.change(dimensionality_reduction, inputs=[df, reduction_choices, df_reduced],
                        #     outputs=[df_reduced])
                        clusters_visualized = gr.Plot()
                        # Clustering Visualization

            # -> UI logic
            # (1) Show plot if dataset_dim ==2 else: show reduction options
            best_config_index_dropdown.change(on_best_cvi_change,
                                              inputs=[best_config_index_dropdown, best_config_per_cvi, df_reduced, df,
                                                      df_needs_reduction, reduction_choices],
                                              outputs=[best_config, df_reduced, clusters_visualized])
    # <-----------------------------Repository --------------------------------------------------------->
    with gr.Tab('(3) Repository'):
        no_datasets_title = gr.Markdown(
            f"<h2 style='text-align: center; color:#3ebefe;'>Number of Datasets in The Repository: {no_datasets}</h2>")

        with gr.Accordion("Trained Meta-Learners", elem_id="my_accordion"):
            gr.Markdown("""
            A list of all the trained meta-learners along with their meta-data. 
            """)
            meta_learners_df  = gr.Dataframe(mldf, elem_id="small")

        with gr.Accordion("Train Meta Learner", elem_id="my_accordion"):
            gr.Markdown("""To train a new meta-learner please configure the following and press the "train button".""")
            train_ml_btn = gradio.Button("Train")
            with gr.Row():
                with gr.Column(elem_id="border_col"):
                    gr.Markdown("""(A) Configure The Meta Learner Algorithm""")
                    ml_select = gr.Radio(["KNN", "DT"], label="Select Classifier", value="KNN")
                    knn_options = {"no_neighbors": gr.Slider(2, 10, value=5, step=1, interactive=True,
                                                             label="Number of Neighbors", visible=True),
                                   "metric": gr.Radio(choices=["euclidean", "l1", "cosine"], label='metric',
                                                      visible=True, interactive=True, value="euclidean")
                                   }
                    dt_options = {"criterion": gr.Radio(choices=["gini", "entropy", "log_loss"], label='metric',
                                                        visible=False, interactive=True)
                                  }
                    ml_select.change(fn=update_ml_options, inputs=ml_select,
                                     outputs=[knn_options[key] for key in knn_options] +
                                             [dt_options[key] for key in dt_options])

                with gr.Column(elem_id="border_col"):
                    gr.Markdown("""(B) Select The meta-features group to include as training variables""")

                    mf_selection = gr.Radio(["All", "Custom Selection"], value="All")
                    mf_search_type = gr.Radio(["Category", "Paper"], label="Search Type", visible=False)
                    mf_search_choices = gr.Dropdown(choices=[], multiselect=True, interactive=True, visible=False)

                    # On change effects
                    mf_selection.change(fn=toggle_mf_selection_options, inputs=[mf_selection],
                                        outputs=[mf_search_type, mf_search_choices])

                    mf_search_type.change(update_search_type_dropdown, inputs=mf_search_type,
                                          outputs=mf_search_choices)

                with gr.Column(elem_id="border_col"):
                    gr.Markdown("""(C) Select how to appoint the best algorithm to each dataset""")
                    best_alg_selection = gr.Radio(["Most Popular Alg", "Specific CVI"], value="Most Popular Alg")
                    best_config_ml = gr.Dropdown(choices=cvi_list, multiselect=False,
                                                 label="Select Index", visible=False, interactive=True)

                    best_alg_selection.change(
                        lambda x: gr.update(visible=True) if x == "Specific CVI" else gr.update(visible=False),
                        inputs=best_alg_selection, outputs=best_config_ml)

        with gr.Accordion("Meta-Learner Results", elem_id="my_accordion"):
            with gr.Row():
                ml_id = gr.Textbox(label="Provide meta learner ID", interactive=True)
                ml_add_to_repo_btn = gr.Button("Add meta-learner to repository ")

            with gr.Row():
                ml_cm_img = gr.Image()
                ml_meta_data = gr.JSON()

        train_ml_btn.click(train_meta_learner,
                           inputs=[ml_select, mf_selection, best_alg_selection] +
                                  [knn_options[x] for x in knn_options.keys()] +
                                  [dt_options[x] for x in dt_options] + [mf_search_type, mf_search_choices] +
                                  [best_config_ml],
                           outputs=[ml_cm_img, ml_meta_data, meta_learner_trained])

        ml_add_to_repo_btn.click(save_meta_learner, inputs=[meta_learner_trained, ml_id, ml_meta_data],
                                 outputs = [meta_learners_df, ml_model_choice])

    add_data_to_repo_btn.click(on_add_data_to_repo, inputs=[data_id, df, mf_calculated, best_config_per_cvi],
                               outputs=[add_data_to_repo_title, no_datasets_title])
    df.change(on_df_load, inputs=df, outputs=[df_needs_reduction, model_search_tab, clustering_exploration_tab,
                                              not_loaded_message])

    df_needs_reduction.change(df_needs_dimred, inputs=df_needs_reduction, outputs=reduction_column)

    operations_complete.change(on_operations_change, inputs=[operations_complete],
                               outputs=[demo_mf_completed, demo_ms_completed, title_row_2, add_data_to_repo_title,
                                        add_data_to_repo_btn])

# Create repo folders for meta-features and es results
cwd = os.getcwd()
if not os.path.isdir(os.path.join(cwd, "results")):
    os.mkdir(os.path.join(cwd, "results"))

mf_path = os.path.join(cwd, "results", "mf")
es_path = os.path.join(cwd, "results", "es")

if not os.path.isdir(mf_path):
    os.mkdir(mf_path)

if not os.path.isdir(es_path):
    os.mkdir(es_path)

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("../repository", exist_ok=True)
    os.makedirs("../repository/best_alg_per_cvi", exist_ok=True)
    os.makedirs("../repository/meta_features", exist_ok=True)
    os.makedirs("../repository/meta_learners", exist_ok=True)
    os.makedirs("../repository/trials", exist_ok=True)
    os.makedirs("../repository/datasets", exist_ok=True)
    demo.launch(server_name="127.0.0.1",share=False,debug=True, server_port=7861)