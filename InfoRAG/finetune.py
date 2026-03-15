import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, TextStreamer, DataCollatorForLanguageModeling
from trl import DataCollatorForCompletionOnlyLM
import pandas as pd
from transformers import TrainerCallback
from datetime import datetime
import gc

# Loading unsloth models wrapper
def load_unsloth(model_name, max_seq_length):
    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=f"unsloth/{model_name}",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
    )
    return model, tokenizer

# Clearing memory
def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

class LoggingCallback(TrainerCallback):
    def __init__(self, logs_dict):
        self.logs_dict = logs_dict
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                step = state.global_step
                loss = logs.get('loss')
                learning_rate = logs.get('learning_rate')
                timestamp = datetime.now().isoformat()
                self.logs_dict['step'].append(step)
                self.logs_dict['loss'].append(loss)
                self.logs_dict['learning_rate'].append(learning_rate)
                self.logs_dict['timestamp'].append(timestamp)
                print(f"[Step {step:6d}] Loss: {loss:.4f} | LR: {learning_rate:.2e} | Time: {timestamp}")

class finetuneLLM():
    def __init__(
        self, model_name, data, max_seq_length=1024,
        
        r=16, lora_alpha=16, lora_dropout=0, use_rslora=True, 

        dataset_num_proc=2, packing=False, learning_rate=1e-5, lr_scheduler_type="linear", per_device_train_batch_size=2, gradient_accumulation_steps=8, max_steps=10000, fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(), logging_steps=25, optim="paged_adamw_8bit", weight_decay=0.01, warmup_steps=500, seed=0, save_steps=1000, gradient_checkpointing=True, report_to=['tensorboard'], push_to_hub=True,
                ):

        # Initialize parameters
        self.model_name = model_name
        self.data = data
        self.logs = {
            'step': [],
            'loss': [],
            'learning_rate': [],
            'timestamp': [],
        }
        self.hyperparameters = {
            'lora': {
                'max_seq_length': max_seq_length,
                'r': r,
                'lora_alpha': lora_alpha,
                'lora_dropout': lora_dropout,
                'use_rslora': use_rslora,   
            },
            'trainer': {
                'max_seq_length': max_seq_length,
                'dataset_num_proc': dataset_num_proc,
                'packing': packing,
                'learning_rate': learning_rate,
                'lr_scheduler_type': lr_scheduler_type,
                'per_device_train_batch_size': per_device_train_batch_size,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'max_steps': max_steps,
                'fp16': fp16,
                'bf16': bf16,
                'logging_steps': logging_steps,
                'optim': optim,
                'weight_decay': weight_decay,
                'warmup_steps': warmup_steps,
                'seed': seed,
                'save_steps': save_steps,
                'gradient_checkpointing': gradient_checkpointing,
                'report_to': report_to,
                'push_to_hub': push_to_hub,
            }
        }

        # Load stuff in 
        self.model, self.tokenizer = load_unsloth(model_name=model_name, max_seq_length=max_seq_length)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[SUFFIX]"]})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self._load_dataset()
        self._load_lora()
        self._load_collator()
        self._load_trainer()

        print("Done initializing, run .script in order to finetune")

    def _load_dataset(self):
        if isinstance(self.data, str):
            info_data = pd.read_csv(self.data)
        elif isinstance(self.data, pd.DataFrame):
            info_data = self.data
        else:
            print("Unrecognized data format, please use string or dataframe")
            return
        info_data['text'] = (
            "Context: " + info_data['context_clean'] + 
            "\nPrefix: " + info_data['prefix_clean'] + 
            "[SUFFIX]" + info_data['suffix_clean']
        )
        dataset = Dataset.from_dict({
            'text': info_data['text'].astype(str).tolist()
        })
        self.dataset = dataset

    def _load_lora(self):
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.hyperparameters['lora']['r'],
            lora_alpha=self.hyperparameters['lora']['lora_alpha'],
            lora_dropout=self.hyperparameters['lora']['lora_dropout'],
            use_rslora=self.hyperparameters['lora']['use_rslora'],
            use_gradient_checkpointing="unsloth"
        )

    def _load_collator(self):
        suffix_token_id = self.tokenizer.convert_tokens_to_ids("[SUFFIX]")
        self.collator = DataCollatorForCompletionOnlyLM(
            response_template=[suffix_token_id],
            tokenizer=self.tokenizer
        )
        test = self.tokenizer("Context: foo\nPrefix: bar[SUFFIX]baz")
        labels = self.collator.torch_call([test])
        assert (labels['labels'][0] == -100).sum() > 0, "Collator not masking context!"
        assert (labels['labels'][0] != -100).sum() > 0, "Collator not training on any tokens - template not found!"

    def _load_trainer(self):
        self.trainer = SFTTrainer(
            model=self.model,
            data_collator=self.collator,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            dataset_text_field="text",
            max_seq_length=self.hyperparameters['trainer']['max_seq_length'],
            dataset_num_proc=self.hyperparameters['trainer']['dataset_num_proc'],
            packing=self.hyperparameters['trainer']['packing'],
            callbacks=[LoggingCallback(self.logs)],
            args=TrainingArguments(
                learning_rate=self.hyperparameters['trainer']['learning_rate'],
                lr_scheduler_type=self.hyperparameters['trainer']['lr_scheduler_type'],
                per_device_train_batch_size=self.hyperparameters['trainer']['per_device_train_batch_size'],
                gradient_accumulation_steps=self.hyperparameters['trainer']['gradient_accumulation_steps'],
                max_steps=self.hyperparameters['trainer']['max_steps'],
                fp16=self.hyperparameters['trainer']['fp16'],
                bf16=self.hyperparameters['trainer']['bf16'],
                logging_steps=self.hyperparameters['trainer']['logging_steps'],
                optim=self.hyperparameters['trainer']['optim'],
                weight_decay=self.hyperparameters['trainer']['weight_decay'],
                warmup_steps=self.hyperparameters['trainer']['warmup_steps'],
                output_dir=f"./{self.model_name}-InfoRAG",
                seed=self.hyperparameters['trainer']['seed'],
                save_steps=self.hyperparameters['trainer']['save_steps'],
                gradient_checkpointing=self.hyperparameters['trainer']['gradient_checkpointing'],
                report_to=self.hyperparameters['trainer']['report_to'],
                push_to_hub=self.hyperparameters['trainer']['push_to_hub'],
            ),
        )
        
    def script(self):
        try:
            print("Now training:", self.model_name)
            self.trainer.train()
            kwargs = {
                "dataset_tags": "wikimedia/wikipedia",
                "dataset": "psgs_w100",
                "dataset_args": "config: wikipedia, split: train",
                "language": "en",
                "model_name": f"{self.model_name} InfoRag - Chengyi Li",
                "finetuned_from": f"unsloth/{self.model_name}",
                "tasks": "causal-language-modeling",
            }
            self.trainer.push_to_hub(**kwargs)
            self.logs = pd.DataFrame(self.logs)
            self.logs.to_csv(f"{self.model_name+'-InfoRAG'}/{self.model_name}_logs.csv")
            
        finally:
            del self.trainer
            del self.model
            del self.tokenizer
            del self.dataset
            clear_gpu_memory()
