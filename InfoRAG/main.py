# Libraries for finetuning that need to be imported in a certain order
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, TextStreamer, DataCollatorForLanguageModeling, pipeline, BitsAndBytesConfig
import pandas as pd
from transformers import TrainerCallback, AutoTokenizer
import gc
from peft import PeftModel

# This is the main script that runs the full project
from make_data import create_info_data
from finetune import finetuneLLM, load_unsloth
from evaluation_pipeline import Evaluator

# Preprocess evaluation set function
def preprocess_hotpot(example):
    context_parts = []
    for title, sentences in zip(example['context']['title'], example['context']['sentences']):
        context_parts.append(f"{title}: {' '.join(sentences)}")
    return {
        'question' : example['question'],
        'answer': example['answer'],
        'context': ' '.join(context_parts)
    }
def preprocess_wow(example):
    return {
        'context': example['persona'],
        'question': "You have just met the other person, who seems quite curious, and you are eager to discuss a topic with them!", # https://ar5iv.labs.arxiv.org/html/1811.01241 I got the prompt from here
        'answer': example['text']
    }

# Clearing memory
def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

if __name__ == "__main__":

    # Clear cache
    print("Script START")
    clear_gpu_memory()

    # Initialize variables
    unsloth_models = {"Qwen2.5-3B":[], "Qwen2.5-1.5B":[], "Qwen2.5-0.5B":[], 
                      "Llama-3.2-3B":[], "Llama-3.2-1B":[], 
                      "gemma-2-2b":[], "gemma-2b":[], 
                      "phi-2":[], "SmolLM2-1.7B":[]}
    max_steps = 5000
    max_samples = 300
    
    # Set to True if done already
    data_is_made = True
    finetuned = False
    evaluated = False
    
    # Making dataset
    if not data_is_made:
        info_data = create_info_data(input_path="psgs_w100.tsv", output_path="dataset.csv", subset=500000)
        clear_gpu_memory()
    else:
        info_data = pd.read_csv("dataset.csv")
    
    # Finetune models
    if not finetuned:
        for model in unsloth_models.keys():
            try:
                print("Finetuning:", model)
                finetune = finetuneLLM(model_name=model, data=info_data, max_steps=max_steps)
                finetune.script()
                del finetune
                clear_gpu_memory()
            except Exception as e:
                print(f"Failed to finetune {model}: {e}")
            clear_gpu_memory()

    # Evaluate
    if not evaluated:
        hotpot_dataset = load_dataset('hotpotqa/hotpot_qa', 'distractor', split='validation') # hotpot only has validation in HF
        wow_dataset = load_dataset('Organika/wizard_of_wikipedia', split='test')
        hotpot_template = "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        hotpot_evaluator = Evaluator(
            prompt_template=hotpot_template,
            question_key='question',
            answer_key='answer',
            context_key='context'
        )
        wow_template = "Context: {context}\n\nConversation: {question}\n\nAnswer:"
        wow_evaluator = Evaluator(
            prompt_template=wow_template,
            question_key='question',
            answer_key='answer',
            context_key='context'
        )
        for model_name in unsloth_models.keys():
            
            # Base model evaluate
            print(f"Loading base model: {model_name}")
            base_model, base_tokenizer = FastLanguageModel.from_pretrained(
                model_name = f"unsloth/{model_name}",
                load_in_4bit = True,
            )
            base_tokenizer.add_special_tokens({"additional_special_tokens": ["[SUFFIX]"]})
            base_model.resize_token_embeddings(len(base_tokenizer))
            FastLanguageModel.for_inference(base_model)
            base_llm = pipeline(
                task="text-generation", 
                model=base_model, 
                tokenizer=base_tokenizer
            )
            hotpot_evaluator.evaluation(llm=base_llm, dataset=hotpot_dataset, model_name=f'base-{model_name}-hotpot', preprocess_fn=preprocess_hotpot, max_samples=max_samples)
            wow_evaluator.evaluation(llm=base_llm, dataset=wow_dataset, model_name=f'base-{model_name}-wow', preprocess_fn=preprocess_wow, max_samples=max_samples)

            # Cleanup
            del base_llm
            del base_model
            clear_gpu_memory()

            # Evaluate checkpoints
            for i in range(1, (max_steps // 1000) + 1):
                step = f"{i}000"
                checkpoint_path = f"{model_name}-InfoRAG/checkpoint-{step}"
                print(f"Loading checkpoint: {checkpoint_path}")
                try:
                    tokenizer_ckpt = AutoTokenizer.from_pretrained(checkpoint_path)
                    model_ckpt, __ = FastLanguageModel.from_pretrained(
                        model_name = f"unsloth/{model_name}",
                        load_in_4bit = True,
                    )
                    model_ckpt.resize_token_embeddings(len(tokenizer_ckpt))
                    model_ckpt = PeftModel.from_pretrained(model_ckpt, checkpoint_path)
                    FastLanguageModel.for_inference(model_ckpt)
                    tuned_llm = pipeline(task="text-generation", model=model_ckpt, tokenizer=tokenizer_ckpt)
                    hotpot_evaluator.evaluation(llm=tuned_llm, dataset=hotpot_dataset, model_name=f'tuned-{model_name}-{step}-hotpot', preprocess_fn=preprocess_hotpot, max_samples=max_samples)
                    wow_evaluator.evaluation(llm=tuned_llm, dataset=wow_dataset, model_name=f'tuned-{model_name}-{step}-wow', preprocess_fn=preprocess_wow, max_samples=max_samples)
                    
                    # Cleanup
                    del tuned_llm
                    del model_ckpt
                    del tokenizer_ckpt
                    clear_gpu_memory()
                    
                except Exception as e:
                    print(f"Failed to load checkpoint {step}: {e}")

            # Save results for this model family
            hotpot_evaluator.display_results()
            wow_evaluator.display_results()
            pd.DataFrame(hotpot_evaluator.results).to_csv(f"{model_name}-InfoRAG/{model_name}_hotpot_final.csv")
            pd.DataFrame(wow_evaluator.results).to_csv(f"{model_name}-InfoRAG/{model_name}_wow_final.csv")
            
            # Clear cache
            clear_gpu_memory()

    # Finish, clear cache again
    print("Script END")
    clear_gpu_memory()
        