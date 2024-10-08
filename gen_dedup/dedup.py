# -*- coding: utf-8 -*-

import os
import json
from typing import Optional, Dict
from collections import defaultdict

import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    AutoTokenizer, DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)

from .models.modeling_t5 import T5ForGenerativeDeduplication


ALL_SUPPORT_GEN_MODELS = {
    'T5ForGenerativeDeduplication': T5ForGenerativeDeduplication,
}


class GenDedup:
    def __init__(self,
                 model_name_or_path: str,
                 model_class_name: str = 'T5ForGenerativeDeduplication',
                 pretrained_model_name_or_path: Optional[str] = None,
                 max_length: Optional[int] = None,
                 gaussian_noise_prob: float = 0.1,
                 gaussian_noise_variance: float = 0.1):
        model_class = ALL_SUPPORT_GEN_MODELS[model_class_name]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = model_class.from_pretrained(pretrained_model_name_or_path or model_name_or_path)
        self.model.setup_gd(
            gaussian_noise_prob=gaussian_noise_prob,
            gaussian_noise_variance=gaussian_noise_variance)
        self.max_length = max_length
        self.ds = None
        self.feature_columns = self.tokenizer('gen dedup').keys()

    def prepare_ds(self, ds: Dataset) -> Dataset:
        def preprocess_function(examples):
            inputs = [d for d in examples["sentence"]]
            targets = examples['labels']
            model_inputs = self.tokenizer(inputs, max_length=self.max_length, truncation=True)
            labels = self.tokenizer(text_target=targets, max_length=self.max_length, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            model_inputs["label_text"] = examples['labels']
            return model_inputs

        return ds.map(preprocess_function, batched=True)

    def fit(self,
            ds: Dataset,
            output_dir: str,
            batch_size: int = 32,
            epochs: int = 1,
            learning_rate: float = 1e-4,
            warmup_steps: int = 0,
            logging_steps: int = 10,
            weight_decay: float = 0.01,
            gradient_accumulation_steps: int = 1,
            fp16: Optional[bool] = None,
            argument_kwargs: Optional[Dict] = None,
            trainer_kwargs: Optional[Dict] = None):
        self.ds = self.prepare_ds(ds)
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)

        if argument_kwargs is None:
            argument_kwargs = {}
        if trainer_kwargs is None:
            trainer_kwargs = {}
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            weight_decay=weight_decay,
            save_total_limit=1,
            num_train_epochs=epochs,
            logging_steps=logging_steps,
            warmup_steps=warmup_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            fp16=fp16,
            predict_with_generate=True,
            prediction_loss_only=True,
            **argument_kwargs
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.ds,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
        self.model.eval()

    def deduplicate(self,
                    save_dir: str,
                    max_label_words: int = 1,
                    ds: Optional[Dataset] = None,
                    threshold: float = 0.5,
                    device: Optional[str] = None,
                    generate_kwargs: Optional[Dict] = None,
                    exist_ok: bool = True) -> Dict:
        if generate_kwargs is None:
            generate_kwargs = {}

        os.makedirs(save_dir, exist_ok=exist_ok)

        self.model = self.model.eval()
        if device is None:
            device = self.model.device

        if ds is not None:
            ds = self.prepare_ds(ds)
        else:
            ds = self.ds

        duplicate_map = defaultdict(list)
        non_duplicates = list()
        for obj in tqdm(ds):
            inputs = {}
            for name in self.feature_columns:
                inputs[name] = torch.LongTensor([obj[name]]).to(device)
            outputs = self.model.generate(
                **inputs,
                output_scores=True,
                return_dict_in_generate=True,
                **generate_kwargs)
            predict_ids = outputs.sequences[0]
            gen_label = self.tokenizer.decode(predict_ids, skip_special_tokens=True)
            gen_label = ' '.join(gen_label.split()[:max_label_words])
            scores = torch.nn.functional.softmax(outputs.scores[0], dim=-1)
            raw_obj = {k: v for k, v in obj.items() if k not in self.feature_columns}
            if gen_label == obj['label_text']:
                if threshold is not None and any(scores[0, i] for i in predict_ids) < threshold:
                    continue
                duplicate_map[obj['label_text']].append(raw_obj)
            else:
                non_duplicates.append(raw_obj)

        duplicate_total = 0
        nonduplicate_total = 0
        duplicate_path = os.path.join(save_dir, 'duplicate.jsonl')
        nonduplicate_path = os.path.join(save_dir, 'nonduplicate.jsonl')
        with open(duplicate_path, 'w') as writer, open(nonduplicate_path, 'w') as writer2:
            for obj in non_duplicates:
                writer2.writelines(json.dumps(obj, ensure_ascii=False) + '\n')
                nonduplicate_total += 1

            for label, obj_list in duplicate_map.items():
                writer2.writelines(json.dumps(obj_list[0], ensure_ascii=False) + '\n')
                nonduplicate_total += 1
                for obj in obj_list[1:]:
                    duplicate_total += 1
                    writer.writelines(json.dumps(obj, ensure_ascii=False) + '\n')
           
        print(f'{duplicate_total} duplicate text detected!')
        print(f'{nonduplicate_total} nonduplicate text detected!')
        print(f'Duplicate text has been saved to {duplicate_path}')
        print(f'Nonduplicate text has been saved to {nonduplicate_path}')
