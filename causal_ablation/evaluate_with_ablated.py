import json
import argparse
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm
import torch
from transformer_lens import HookedEncoderDecoder
import transformer_lens.utils as utils
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
import re
from ablation import evaluate, ablation_hook_factory, add_logit_gt, get_doc_to_logit, filter_on_topic
import logging

def load_ablated_neurons(file_path):
    with open(file_path, 'r') as f:
        ablated_neurons = json.load(f)
    return ablated_neurons

def main():
    parser = argparse.ArgumentParser(description='Evaluate ablated model on eval_set set')
    parser.add_argument('--file', type=str, required=True, help='Path to ablated_neurons.json')
    parser.add_argument('--checkpoint', type=str, default=r'C:\projects\transformers\236004-HW1-GPT\DSI-large-7423', help='Model checkpoint directory')
    parser.add_argument('--data_path', type=str, default=r'C:\projects\transformers\236004-HW1-GPT\NQ10k', help='Path to data directory')
    parser.add_argument('--chunk_size', type=int, default=10, help='Batch size for evaluation')
    parser.add_argument('--validation', action='store_true', help='Use validation set instead of the test set to evaluate on')
    parser.add_argument('--no_ablations', action='store_true', help='Skip ablation and evaluate the original model')
    parser.add_argument("--topics_path", type=str, default="C:\\Users\\Administrator\\OneDrive\\Documents\\university\\transformers",
                        help='path to topics.pkl, selected.npy (expected in the same path)')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.CRITICAL)

    # Load eval_set set
    with open(os.path.join(args.data_path, 'documents-10000-7423.json')) as fp:
        docs = json.load(fp)
    if args.validation:
        with open(os.path.join(args.data_path, 'val_queries-10000-7423.json')) as fp:
            eval_set = json.load(fp)
    else:
        with open(os.path.join(args.data_path, 'test_queries-10000-7423.json')) as fp:
            eval_set = json.load(fp)
    path_dir = os.path.dirname(args.file)
    topic_path = os.path.join(path_dir,'topic.json')
    if os.path.exists(topic_path):
        with open(topic_path) as fp:
            topic = json.load(fp)
        print(topic)
        topic_id = topic['topic_id']
        eval_set = filter_on_topic(eval_set, docs, args.topics_path, 0, None, topic_id)

    if not args.no_ablations:
        ablated_neurons = load_ablated_neurons(args.file)
        print(f"Loaded {len(ablated_neurons)} ablated neurons from {args.file}")
    else:
        ablated_neurons = []
        print("Skipping ablation - evaluating original model")

    # Model and tokenizer loading
    OFFICIAL_MODEL_NAMES.append(args.checkpoint)
    hf_model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    device = str(utils.get_device())
    model = HookedEncoderDecoder.from_pretrained(args.checkpoint, hf_model=hf_model, device=device)

    # Token id mapping for docid
    tokenizer_t5 = AutoTokenizer.from_pretrained('google-t5/t5-large')
    first_added_doc_id = len(tokenizer_t5)
    last_added_doc_id = len(tokenizer_t5) + (len(tokenizer) - len(tokenizer_t5))
    tokens_by_tokenization_order = tokenizer.batch_decode(np.arange(first_added_doc_id, last_added_doc_id))

    doc_to_logit, first_added_doc_id, last_added_doc_id = get_doc_to_logit(tokenizer, tokenizer_t5)
    del tokenizer_t5

    add_logit_gt(eval_set, doc_to_logit)
    # add_logit_gt(eval_set, doc_to_logit) # This line is removed as per the edit hint

    # Register ablation hooks for all ablated neurons
    if not args.no_ablations and ablated_neurons:
        # Group by layer for efficiency
        from collections import defaultdict
        layer_to_neurons = defaultdict(list)
        for layer, neuron_idx in ablated_neurons:
            layer_to_neurons[layer].append(neuron_idx)
        for layer, neuron_indices in layer_to_neurons.items():
            hook_name = f"decoder.{layer}.mlp.hook_post"
            for neuron_idx in neuron_indices:
                model.hook_dict[hook_name].add_hook(ablation_hook_factory(neuron_idx))
        print(f"Registered ablation hooks for {len(ablated_neurons)} neurons.")
    else:
        print("No ablation hooks registered - using original model.")

    # Evaluate
    print("Evaluating ablated model on eval_set set...")
    logits, mrr, hits_at_1, hits_at_5, hits_at_10, _, _ = evaluate(eval_set, args.chunk_size, tokenizer, model, device)
    print(f"Ablated Model Results on eval_set Set:")
    print(f"  HITS@1: {hits_at_1:.4f}")
    print(f"  HITS@5: {hits_at_5:.4f}")
    print(f"  HITS@10: {hits_at_10:.4f}")
    if mrr is not None:
        print(f"  MRR: {mrr:.4f}")

    out_filename = ('v_' if args.validation else 't_') + 'eval.json'
    with open(os.path.join(path_dir, out_filename), 'w') as fp:
        json.dump({'hits_at_1': hits_at_1},fp)

if __name__ == "__main__":
    main()
