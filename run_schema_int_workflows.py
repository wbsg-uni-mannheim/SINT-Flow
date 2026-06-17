from model import SchemaIntegrationRuns
import os
from dotenv import dotenv_values
from langchain.chat_models import init_chat_model
import pdb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

if __name__ == "__main__":
    # Initialize LLM
    model_name = "gpt-5.2-2025-12-11"
    # model_name = "Qwen/Qwen3.6-27B"
    reasoning = "None"
    model_type = "openai"
    # model_type = "hf" # For models loaded through HuggingFace
    
    if model_type == "hf":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16"
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="hf_cache/")
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto", cache_dir="hf_cache/", quantization_config=quant_config)
        if "qwen" in model_name.lower():
            # Update chat format to fix issues with tool calls
            with open("preprocessing/qwen_chat_template.jinja", "r") as f:
                tokenizer.chat_template = f.read()
    else:
        config = dotenv_values("../key.env")
        os.environ['OPENAI_API_KEY'] = config["OPENAI_API_KEY"]
        OPENAI_API_KEY = config["OPENAI_API_KEY"]
        
        if reasoning != "None":
            model = init_chat_model(openai_api_key=OPENAI_API_KEY, temperature=1, model=model_name, reasoning = {"effort": reasoning, "summary": "auto"})
        else:
            model = init_chat_model(openai_api_key=OPENAI_API_KEY, temperature=1, model=model_name)
        tokenizer = None
    
    # Specify prompt name and benchmark
    prompt_name = "long-prompt-updated-4"
    # benchmark = "Real Benchmark"
    benchmark = "SINT-Benchmark"

    # Run workflows for each use case
    for folder_name in os.listdir(f"data/selected-tables/{benchmark}/"):
        for sequence_of_phases in [
            ["detect_tables_phase", "schema_matching_phase", "grouping_phase", "schema_integration_phase", "final_integration_phase"], # Workflow 1 in the paper
            # ["detect_tables_phase", "grouping_phase", "schema_integration_phase", "final_integration_phase"], # Workflow 2 in the paper
            # ["schema_matching_phase", "schema_integration_phase", "detect_tables_phase", "final_integration_phase"], # Workflow 3 in the paper
            # ["schema_matching_phase"] # Run only schema matching operator for baseline comparison
        ]:
            # Run config
            config = {
                "benchmark": benchmark,
                "folder_name": folder_name,
                "prompt_name": prompt_name,
                "table_path": f"data/selected-tables/{benchmark}/{folder_name}/",
                "table_format": "markdown",
                "table_rows": 10, # For Qwen we use 5
                "model": model,
                "model_type": model_type,
                "model_name": model_name,
                "reasoning": reasoning,
                "tokenizer": tokenizer,
                "demonstration": 0,
                "self_consistency": True,
                # "self_consistency": False,
                "schema_matching_batch_size": 1,
                "merging_batch_size": "all", 
                "num_runs": 3,
                "sequence_of_phases": sequence_of_phases,
            }
    
            schema_integration_workflow = SchemaIntegrationRuns(**config)
            schema_integration_workflow.initialize_workflow()
            schema_integration_workflow.run_workflow()
            # pdb.set_trace()
