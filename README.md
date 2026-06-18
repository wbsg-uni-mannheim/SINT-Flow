### Initialize environment

`conda create --name schema-integration python=3.11` <br>
`pip install -r requirements.txt`


### Prompt Placeholders:
The prompts used in the experiments are read from the 'prompts/prompts.json' file. There are several placeholders used to input the correct information in the prompts. The following placeholders are used in the prompts:

In "Table Splitting" prompts: [TABLE_INPUT], to show the test table to be split <br>
In "Table Grouping" prompts: [TABLES], to show the split tables which will be grouped <br>
In "Attribute Merging" prompts: [TABLES], to show tables from the same group whose attributes are going to be merged <br>
In "Final Schema Integration Output" prompts: [SCHEMATA], to show the schemata of all tables (entity types) integrated in the last phase <br>

### Running the schema integration workflows:
There are five operators that can be arranged in different sequences to form schema integration workflows. The following sequences are tested in our paper:

1. <b>Entity Type Detection on Input Tables</b>: Table Splitting Operator &rarr; Schema Matching Operator &rarr; Table Grouping Operator &rarr; Attribute Merging Operator &rarr; Final Schema Integration Output Operator.

2. <b>Implicit Schema Matching Operator</b>: Table Splitting Operator &rarr; Table Grouping Operator &rarr; Attribute Merging Operator &rarr; Final Schema Integration Output Operator.

3. <b>Entity Type Detection on Integrated Table</b>: Schema Matching Operator &rarr; Attribute Merging Operator &rarr; Table Splitting Operator &rarr; Final Schema Integration Output Operator.

To run the workflows the run_schema_int_workflow.py file can be used. The file contains the code for running the three workflows mentioned above. The file contains the possibility of running the workflows using an OpenAI model or a model accessed through the Huggingface library.

The results of the runs are saved in the 'logs' folder, which records information on the run, for example the prompt used, model, if the self-consistency strategy is used, the time taken for each operator to run, etc.

### Evaluate the Workflows:
The evalution of the workflows can be done using the run_evaluation.ipyb notebook. This file contains the possibility of evaluating the LLM workflos as well as the baselines.

### Evaluation Results
The evaluation results are contained in the 'evaluation_json' and 'evaluation_summary' folders. The first contains the detailed results of all runs of a workflow on a use case, while the second contains the Precision, Recall and F1-scores of those runs.

All detailed results are contained in the "aggregated_results_all_cases_mean.json" file in the 'evaluation_json' folder.

### Baselines
The baselines are contained in the 'baselines' folder. The 'baselines' folder contains the results of the baselines on the use cases.

To run ALITE the code should be downloaded here: <a href="https://github.com/northeastern-datalab/alite">ALITE Repository</a>. The instrunctions for setting up ALITE is available in the ALITE repository.

The files 'alite_turl.py' which is used to generated the TURL embeddings and the 'align_integration_ids.py' are modified files from the original ALITE repository. The qwen_embeddings.py file also contains partial code from the ALITE repository as we generate embeddings using ALITE's iterative method. We provide the generated embeddings for both TURL and Qwen in the 'baselines/run-alite/' folder.

For running SI-LLM, we use the prompts provided in the <a href="https://github.com/PierreWoL/SILLM">SI-LLM Repository</a>. We provide the file run_si-llm.py for replicating the SI-LLM baseline.

Regarding the COMA baseline, we use the implementation provided in the <a href="https://github.com/delftdata/valentine">Valentine Repository</a>. We provide the file run-coma-baseline.py for replicating the COMA baseline.