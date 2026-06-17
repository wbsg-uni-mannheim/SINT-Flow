### Initialize environment

`conda create --name schema-integration python=3.11` <br>
`pip install -r requirements.txt`


### Prompt Placeholders:
For phase 1 prompts: [TABLE_INPUT], to show the test table to be split <br>
For phase 2 prompts: [TABLES], to show the split tables which will be grouped <br>
For phase 3 prompts: [TABLES], to show tables from the same group whose attributes are going to be merged <br>
For phase 4 prompts: [SCHEMATA], to show the schemata of all tables (entity types) integrated in the last phase <br>


### Evaluation Results
The evaluation results are contained in the 'evaluation_json' and 'evaluation_summary' folders. The first contains the detailed results of all runs of a workflow on a use case, while the second contains the Precision, Recall and F1-scores of those runs.

### Baselines
The baselines are contained in the 'baselines' folder. The 'baselines' folder contains the results of the baselines on the use cases.

To run ALITE the code should be downloaded here: The instrunctions for creating the ALITE space is available in the ALITE repository.
The files alite_turl and qwen_embeddings should be put in the folder alite-main/codes/align_utilities. 