import os
import argparse
import json
import time
import numpy as np

import torch

from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification, 
    EvalPrediction, 
    TrainingArguments
)

from methods.mixup.utils import MixupAutoModelForSequenceClassification, MixupTrainer
from utils import *
        
if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=0, type=int, help='Seed')
    parser.add_argument('--model', default='bert-base', type=str, help='only bert-base available')
    parser.add_argument('--data_ratio', default='1.0', type=str, help='data ratio')
    parser.add_argument('--data_dir', default='data', type=str, help='data directory')
    parser.add_argument('--dataset', default='ynat', type=str, help='ynat, hateful')
    parser.add_argument('--output_dir', default='checkpoint/', type=str, help='Checkpoint directory/')
    parser.add_argument('--result_dir', default='results/', type=str, help='Result directory/')
    parser.add_argument('--lr', default=5e-5, type=float, help='Learning rate')
    parser.add_argument('--wr', default=0.0, type=float, help='Warm-up ratio')
    parser.add_argument('--wd', default=0.01, type=float, help='Weight decay coefficient')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size [8, 16, 32]')
    parser.add_argument('--total_epochs', default=10, type=int, help='Number of epochs')
    parser.add_argument('--mixup', default=False, type=str2bool, help="Mixup Method")
    parser.add_argument('--label_smoothing', default=0.0, type=float, help="Label smoothing factor")
    parser.add_argument('--p_threshold', default=0.5, type=float, help="MentorNet")
    parser.add_argument('--warmup_period', default=2, type=int, help="MentorNet")

    p_args = parser.parse_args()

    start = time.time()

    set_seed(p_args.seed)
    if not os.path.exists(p_args.result_dir):
        os.makedirs(p_args.result_dir)

    task_to_keys = {
        "ynat": ("title", None),
        "hateful": ("title", "comments")
    }

    sentence1_key, sentence2_key = task_to_keys[p_args.dataset]
    max_sequence_length = 128
    label_column_name = "label"
        
    # Load the dataset
    data_dir = f"{p_args.data_dir}/{p_args.dataset}"
    data_files = {"train": [], "valid": [], "test": []}
    data_files["train"].append(f"{data_dir}/{p_args.data_ratio}/train.json")
    data_files["valid"].append(f"{data_dir}/{p_args.data_ratio}/valid.json")
    data_files["test"].append(f"{data_dir}/test.json")
    datasets = load_dataset(path="json", data_dir=data_dir, data_files=data_files, field='data')

    # Load the metric
    metric = load_metric('./metric.py')

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # Load the pre-trained model
    label_list = []
    label_list = get_label_list(datasets["train"][label_column_name])
    num_labels = len(label_list)

    if p_args.mixup:
        model = MixupAutoModelForSequenceClassification.from_pretrained(f"klue/{p_args.model}", num_labels=num_labels)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(f"klue/{p_args.model}", num_labels=num_labels)
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(f"klue/{p_args.model}")

    def preprocess_function(examples):
        target = ((examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key]))
        
        result = tokenizer(*target, padding=True, max_length=max_sequence_length, truncation=True)

        return result

    preprocessed_datasets = datasets.map(preprocess_function, batched=True)

    train_dataset = preprocessed_datasets['train']
    valid_dataset = preprocessed_datasets['valid']
    test_dataset = preprocessed_datasets['test']

    args = TrainingArguments(
        output_dir=p_args.output_dir, evaluation_strategy='epoch', learning_rate=p_args.lr,
        per_device_train_batch_size=p_args.batch_size, per_device_eval_batch_size=p_args.batch_size,
        num_train_epochs=p_args.total_epochs, weight_decay=p_args.wd, load_best_model_at_end=True, save_strategy='epoch',
        warmup_ratio=p_args.wr, seed=p_args.seed, save_total_limit=1, metric_for_best_model="eval_f1",
        logging_strategy="no", label_smoothing_factor=p_args.label_smoothing, 
        p_threshold=p_args.p_threshold, warmup_period=p_args.warmup_period
    )

    if p_args.mixup:
        trainer = MixupTrainer(
            model, args, train_dataset=train_dataset, eval_dataset=valid_dataset,
            tokenizer=tokenizer, compute_metrics=compute_metrics,
        )
    else:
        trainer = CustomTrainer(
            model, args, train_dataset=train_dataset, eval_dataset=valid_dataset,
            tokenizer=tokenizer, compute_metrics=compute_metrics,
        )

    trainer.train()
    trainer.evaluate()
    predict_result = trainer.predict(test_dataset)

    log_history = trainer.state.log_history

    elapsed_time = (time.time() - start) / 60 # Min.

    log_dir = os.path.join(p_args.result_dir, p_args.dataset)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dirs = os.listdir(log_dir)

    if len(log_dirs) == 0:
        idx = 0
    else:
        idx_list = sorted([int(d.split('_')[0]) for d in log_dirs])
        idx = idx_list[-1] + 1

    from time import strftime

    cur_log_dir = '%d_%s' % (idx, strftime('%Y%m%d-%H%M'))
    full_log_dir = os.path.join(log_dir, cur_log_dir)

    if not os.path.exists(full_log_dir):
        os.mkdir(full_log_dir)

    path = f'{full_log_dir}/model_{p_args.model}_lr_{p_args.lr}_wr_{p_args.wr}_wd_{p_args.wd}_bs_{p_args.batch_size}_te_{p_args.total_epochs}_ls_{p_args.label_smoothing}_mixup_{str(p_args.mixup)}.json'

    with open(path, 'w') as f:
        result = {
            'seed': p_args.seed,
            'time': elapsed_time,
            'train_results': log_history,
            'test_results': predict_result[2],
        }

        json.dump(result, f, indent=2)
