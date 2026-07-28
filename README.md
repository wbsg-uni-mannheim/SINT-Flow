# SINT-Flow: Schema Integration using Large Language Model Workflows

This repository contains the code, data and evaluation results for the paper <a href="https://arxiv.org/abs/2607.24492">**"SINT-Flow: Schema Integration using Large Language Model Workflows"**</a>.

**Paper on arXiv:** <a href="https://arxiv.org/abs/2607.24492">https://arxiv.org/abs/2607.24492</a>

---

The goal of schema integration is, given a set of input schemata or tables, to derive a global, unified schema that is able to represent the concepts, attributes, and relationships of all input tables in a coherent fashion. This paper presents SINT-Flow, a schema integration framework composed of five LLM-based operators that can be combined into workflows to perform fully automated, end-to-end schema integration. To evaluate SINT-Flow, we introduce SINT-Bench, a schema integration benchmark comprising 10 schema integration tasks consisting of altogether 93 relational tables, including tables that describe multiple types of entities.

**Keywords**: Schema Integration, Schema Inference, Schema Management, Schema Integration Benchmark, Large Language Models

## Initialize environment

`conda create --name schema-integration python=3.11` <br>
`pip install -r requirements.txt`

## Running the Schema Integration Workflows:
There are five operators that can be arranged in different sequences to form schema integration workflows. The following sequences are tested in our paper:

1. <b>Entity Type Detection on Input Tables Workflow</b>: Table Splitting Operator &rarr; Schema Matching Operator &rarr; Table Grouping Operator &rarr; Attribute Merging Operator &rarr; Final Schema Integration Output Operator.

2. <b>Implicit Schema Matching Workflow</b>: Table Splitting Operator &rarr; Table Grouping Operator &rarr; Attribute Merging Operator &rarr; Final Schema Integration Output Operator.

3. <b>Entity Type Detection on Integrated Table Workflow</b>: Schema Matching Operator &rarr; Attribute Merging Operator &rarr; Table Splitting Operator &rarr; Final Schema Integration Output Operator.

To run the workflows the [`run_schema_int_workflows.py`](run_schema_int_workflows.py) file can be used. The file contains the code for running the three workflows mentioned above. The file contains the possibility of running the workflows using an OpenAI model or a model accessed through the Huggingface library.

The results of the runs are saved in the [`logs`](logs/) folder, which records information on the run, e.g., the name of the prompt used, name of themodel, if the self-consistency strategy is used, the time in seconds taken for each operator to run, etc.

## Prompt Placeholders:
The prompts used in the experiments are read from the 'prompts/prompts.json' file. There are several placeholders used to input the correct information in the prompts. The following placeholders are used in the prompts:

- In **Table Splitting prompts**: [TABLE_INPUT], to show the test table to be split.
- In **Table Grouping** prompts: [TABLES], to show the split tables which will be grouped.
- In **Attribute Merging** prompts: [TABLES], to show tables from the same group whose attributes are going to be merged.
- In **Final Schema Integration Output** prompts: [SCHEMATA], to show the intermediate integrated schemata of detected entity types.

## SINT-Bench
SINT-Bench is composed of 10 use cases with overall 93 tables, where each table describes 1-3 entity types. The tables of SINT-Bench can be foubnd in the [`data/selected-tables/SINT-Benchmark/`](data/selected-tables/SINT-Benchmark/) folder. The ground truth for all LLM operators can be found in the [`data/gt/SINT-Benchmark/`](data/gt/SINT-Benchmark/) folder. The ground truth is organized in folders for each use case, and is composed overall of 3 files:

1. File ending with `_gt.json`: contains the ground truth for the table splitting operator as well as the table grouping operator.
2. File ending with `_gt_mappings_with_values.json`: contains the ground truth for the attribute merging operator, which includes the mappings of the table columns to integrated attributes for each entity type group, and the schema matching operator ground truth.
3. File ending with `_gt_integration.json`: contains the ground truth for the final schema integration output operator, which includes the final integrated schema for each use case and each entity type detected in the input tables of the use-case.

## Evaluation of the LLM Workflows and Baselines:
The evaluation of the workflows can be done using the [`run_evaluation.ipynb`](run_evaluation.ipynb) notebook. This file contains the possibility of evaluating the LLM workflows as well as the baselines.

## Evaluation Results
The evaluation results are contained in the [`evaluation_json`](evaluation_json/) and [`evaluation_summary`](evaluation_summary/) folders. The first contains the detailed results of all runs of a workflow on a use case, while the second contains the Precision, Recall and F1-scores of those runs.

The aggregated results of all runs of each workflow over all use cases are contained in the [`aggregated_all_cases_mean.json`](evaluation_summary/aggregated_all_cases_mean.json) file in the [`evaluation_summary`](evaluation_summary/) folder.

## Running the Baselines
The code for replicating the baseline results can be found in the [`baselines`](baselines/) folder. We compare the attribute grouping ability of the Schema Matching Operator with three related works:

1. **COMA**: COMA is a well-know label and instance-based schema matching method. To run COMA we use the implementation provided in the <a href="https://github.com/delftdata/valentine">Valentine Repository</a>. To replicate the COMA baseline, we provide the file [`run-coma-baseline.py`](baselines/run-coma-baseline.py).

2. **ALITE**: ALITE is an integration method that embeds columns of input tables and clusters them to find corresponding attributes between tables. ALITE's code and instructions how to run ALITE can be found in the <a href="https://github.com/northeastern-datalab/alite">ALITE Repository</a>. The files `alite_turl.py` which is used to generated the TURL embeddings and the `align_integration_ids.py` in the `baselines` folder are modified files from the original ALITE repository. The `qwen_embeddings.py` file also contains partial code from the ALITE repository as we generate embeddings using ALITE's iterative method. We provide the generated embeddings for both TURL and Qwen in the [`baselines/run-alite/`](baselines/run-alite/) folder.

3. **SI-LLM**: SI-LLM is a schema inference method that uses LLMs to infer hierarchical schemata from input tables. It finds entity types, attributes and relationships between the entity types. To run SI-LLM we use the prompts provided in the <a href="https://github.com/PierreWoL/SILLM">SI-LLM Repository</a>. To replicate the SI-LLM baseline, we provide the file [`run_si-llm.py`](baselines/run_si-llm.py).

## Citation

```bibtex
@misc{korini2026sintflowschemaintegration,
    title={SINT-Flow: Schema Integration using Large Language Model Workflows}, 
    author={Keti Korini and Christian Bizer},
    year={2026},
    eprint={2607.24492},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2607.24492}, 
}
```