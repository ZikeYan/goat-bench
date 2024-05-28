import argparse
import glob
import os
import random
from typing import List
import time
import clip
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer

from goat_bench.utils.utils import load_dataset, load_json, save_pickle, write_json, write_txt

import requests
import spacy


PROMPT = "{category}"


class TaskDescriptionParser:
    def __init__(self):
        # Load the spaCy model
        self.nlp = spacy.load('en_core_web_lg')
        self.words_to_ignore = {'it', 'the', 'left', 'right', 'back', 'front', 'which', 'that', 'side', 'you', 'turn'}
    
    def filter_noun_chunk(self, chunk):
        filtered_words = [token.text for token in chunk if token.text.lower() not in self.words_to_ignore and (token.pos_ == 'NOUN' or token.pos_ == 'PROPN')]
        return " ".join(filtered_words)
    
    def extract_objects_and_adjectives(self, task_description):
        doc = self.nlp(task_description)
        target_object = None
        target_adjectives = []
        other_objects = []

        # Look for the target object and other objects
        for chunk in doc.noun_chunks:
            # Check for adjectives related to the chunk
            adjectives = [token.text for token in chunk if token.pos_ == 'ADJ']

            # Filter the chunk to get the object
            filtered_chunk = self.filter_noun_chunk(chunk)

            if filtered_chunk:
                # Assuming target object is often the first noun or noun phrase
                if target_object is None:
                    target_object = filtered_chunk
                    target_adjectives = adjectives
                else:
                    if filtered_chunk.lower() != target_object.lower():
                        other_objects.append(filtered_chunk)
                    else:
                        target_adjectives.extend(adjectives)

        return target_object, list(set(target_adjectives)), list(set(other_objects))

class BertEmbedder:
    def __init__(self, model: str = "bert-base-uncased") -> None:
        self.model = BertModel.from_pretrained(model)
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.model.to("cuda")

    def embed(self, query: List[str]):
        encoded_dict = self.tokenizer.encode_plus(
            query,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=77,  # Pad & truncate all sentences.
            padding="max_length",
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors="pt",  # Return pytorch tensors.
        ).to("cuda")

        with torch.no_grad():
            outputs = self.model(**encoded_dict)
            embeddings = outputs[0].mean(dim=1).squeeze(0)
        return embeddings

    def batch_embed(self, batch_queries: List[str]):
        embeddings = []
        for query in batch_queries:
            embeddings.append(self.embed(query))
        return embeddings


def tokenize_and_batch(clip, goal_categories):
    tokens = []
    for category in goal_categories:
        prompt = PROMPT.format(category=category)
        tokens.append(clip.tokenize(prompt, context_length=77).numpy())
    return torch.tensor(np.array(tokens)).cuda()


def get_bert():
    bert = BertEmbedder()
    return bert


def save_to_disk(text_embedding, goal_categories, output_path):
    output = {}
    for goal_category, embedding in zip(goal_categories, text_embedding):
        output[goal_category] = embedding.detach().cpu().numpy()
    save_pickle(output, output_path)


def cache_embeddings(goal_categories, output_path, clip_model="RN50"):
    if clip_model == "BERT":
        model = get_bert()
        text_embedding = model.batch_embed(goal_categories)
    else:
        model, _ = clip.load(clip_model)
        batch = tokenize_and_batch(clip, goal_categories)

        with torch.no_grad():
            print(batch.shape)
            text_embedding = model.encode_text(batch.flatten(0, 1)).float()
    print(
        "Goals: {}, Embeddings: {}, Shape: {}".format(
            len(goal_categories), len(text_embedding), text_embedding[0].shape
        )
    )
    save_to_disk(text_embedding, goal_categories, output_path)


def load_categories_from_dataset(dataset_path):
    path = os.path.join(dataset_path, "**/content/*json.gz")
    files = glob.glob(path, recursive=True)

    categories = []
    # for f in tqdm(files):
    #     dataset = load_dataset(f)
    #     for goal_key in dataset["goals_by_category"].keys():
    #         categories.append(goal_key.split("_")[1])
    # return list(set(categories))
    categories = []
    for file in tqdm(files):
        dataset = load_dataset(file)
        for goal_key, goals in dataset["goals"].items():
            for goal in goals:
                if goal.get("object_category") is not None:
                    categories.append(goal["object_category"].lower())
    return list(set(categories))


def clean_instruction(instruction):
    first_3_words = [
        "prefix: instruction: go",
        "instruction: find the",
        "instruction: go to",
        "api_failure",
        "instruction: locate the",
    ]
    for prefix in first_3_words:
        instruction = instruction.replace(prefix, "")
        instruction = instruction.replace("\n", " ")
    # uuid = episode.instructions[0].lower()
    # first_3_words = [
    #     "prefix: instruction: go",
    #     "instruction: find the",
    #     "instruction: go to",
    #     "api_failure",
    #     "instruction: locate the",
    # ]
    # for prefix in first_3_words:
    #     uuid = uuid.replace(prefix, "")
    #     uuid = uuid.replace("\n", " ").strip()
    return instruction.strip()


def cache_ovon_goals(dataset_path, output_path):
    goal_categories = load_categories_from_dataset(dataset_path)
    # val_seen_categories = load_categories_from_dataset(
    #     dataset_path.replace("train", "val_seen")
    # )
    # val_unseen_easy_categories = load_categories_from_dataset(
    #     dataset_path.replace("train", "val_unseen_easy")
    # )
    # val_unseen_hard_categories = load_categories_from_dataset(
    #     dataset_path.replace("train", "val_unseen_hard")
    # )
    # goal_categories.extend(val_seen_categories)
    # goal_categories.extend(val_unseen_easy_categories)
    # goal_categories.extend(val_unseen_hard_categories)

    print("Total goal categories: {}".format(len(goal_categories)))
    # print(
    #     "Train categories: {}, Val seen categories: {}, Val unseen easy categories: {}, Val unseen hard categories: {}".format(
    #         len(goal_categories),
    #         len(val_seen_categories),
    #         len(val_unseen_easy_categories),
    #         len(val_unseen_hard_categories),
    #     )
    # )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cache_embeddings(goal_categories, output_path)


def cache_language_goals(dataset_path, output_path, model):
    files = glob.glob(os.path.join(dataset_path, "*json.gz"))
    instructions = set()
    first_3_words = set()

    filtered_goals = 0
    for file in tqdm(files):
        dataset = load_dataset(file)
        for episode in dataset["episodes"]:
            if "failure" in episode["instructions"][0].lower():
                continue
            cleaned_instruction = clean_instruction(
                episode["instructions"][0].lower()
            )

            if len(cleaned_instruction.split(" ")) > 55:
                filtered_goals += 1
                continue

            instructions.add(cleaned_instruction)
            first_3_words.add(
                " ".join(episode["instructions"][0].lower().split(" ")[:3])
            )

    print(
        "Total instructions: {}, Filtered: {}".format(
            len(instructions), filtered_goals
        )
    )
    max_instruction_len = 0
    for instruction in instructions:
        max_instruction_len = max(
            max_instruction_len, len(instruction.split(" "))
        )
    print("Max instruction length: {}".format(max_instruction_len))
    print("First 3 words: {}".format(first_3_words))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cache_embeddings(list(instructions), output_path, model)


def cache_goat_goals(dataset_path, output_path, model, split):
    # if split !="":
    #     path = os.path.join(dataset_path, f"{split}/content/*json.gz")
    # else:
    #     path = os.path.join(dataset_path, "**/content/*json.gz")
    # files = glob.glob(path, recursive=True)
    # instructions = set()
    # first_3_words = set()
    # filtered_goals = 0
    # for file in tqdm(files):
    #     dataset = load_dataset(file)
    #     for goal_key, goals in dataset["goals"].items():
    #         for goal in goals:
    #             if goal.get("lang_desc") is None:
    #                 continue
    #             cleaned_instruction = goal["lang_desc"].lower()
    #             # cleaned_instruction = clean_instruction(
    #             #     episode["instructions"][0].lower()
    #             # )

    #             if len(cleaned_instruction.split(" ")) > 55:
    #                 filtered_goals += 1
    #                 continue

    #             instructions.add(cleaned_instruction)
    #             first_3_words.add(
    #                 " ".join(cleaned_instruction.lower().split(" ")[:3])
    #             )

    # print("Total goat instructions: {}".format(len(instructions)))
    # max_instruction_len = 0
    # for instruction in instructions:
    #     max_instruction_len = max(
    #         max_instruction_len, len(instruction.split(" "))
    #     )
    # print("Max instruction length: {}".format(max_instruction_len))
    # print("First 3 words: {}".format(first_3_words))
    # write_txt(list(instructions), output_path + f"inst_{split}.txt")
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path + f"inst_{split}.txt", 'r') as file:
        instructions = [line.strip() for line in file.readlines()]

    # cache_embeddings(list(instructions), output_path, model)
    cache_parsed_insts_spacy(list(instructions), output_path, split)

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
headers = {"Authorization": "Bearer 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"}
headers = {"Authorization": "Bearer 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"}
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
def text_to_dict(generated_text):
    try:
        # Convert string to dictionary
        return eval(generated_text.strip())
    except Exception as e:
        print(f"Error parsing generated text: {e}")
        return None
def cache_parsed_insts_spacy(instructions, output_path, split=None):
    parser = TaskDescriptionParser()
    if split != "":
        output_fname = output_path + f"llm_parse_{split}_spacy.json"
    else:
        output_fname = output_path + "llm_parse_spacy.json"
    results = {}
    for inst in sorted(instructions):
        inst = inst.strip()
        target, adjectives, objects = parser.extract_objects_and_adjectives(inst)
        results[inst] = {
            "target":target,
            "adjs":adjectives,
            "nearby":objects
        }
    write_json(results, output_fname)
    


def cache_parsed_insts(instructions, output_path, split=None):
    if split != "":
        output_fname = output_path + f"llm_parse_{split}.json"
        fail_fname = output_path + f"fail_{split}.json"
    else:
        output_fname = output_path + "llm_parse.json"
        fail_fname = output_path + f"fail.json"
    base_prompt = """
    We are performing navigation based on language descriptions. Please parse and strictly reply with this template without any other content:

    {
    "target": "target object within the instruction",
    "adjs": ["adjectives used for describing the target"],
    "nearby": [("phrase describing position relationship", "nearby object")]
    }

    Here are the language descriptions:
    """
    try:
        results = load_json(output_fname)
        fails = load_json(fail_fname)
        cleaned = load_json(output_path + f"cleaned_{split}.json")
    except Exception as e:
        results = {}
        fails = {}
    max_try = 1
    print(len(list(results.keys())+list(fails.keys())))
    print(len(set(list(results.keys())+list(cleaned.keys()))))
    cnt = 0
    for k in fails.keys():
        if k not in cleaned.keys():
            cnt+=1
            print(k)
    print(cnt)
    quit()
    for inst in sorted(instructions):
        inst = inst.strip()
        if inst in results.keys() or inst in fails.keys():
            continue
        for attempt in range(max_try):
            prompt = base_prompt + inst
            print(f"Attempt {attempt + 1} for instruction: {inst}")
            
            output = query({
                "inputs": prompt,
                "parameters": {
                    "max_length": 4096,  # Adjust this based on the model's limits
                    "temperature": 0.001,
                    "return_full_text": False  # Do not return the input in the output
                }
            })
            generated_text = output[0]['generated_text']
            parsed_result = text_to_dict(generated_text)
            
            if parsed_result:
                results[inst] = parsed_result
                write_json(results, output_fname)
                break  # Exit the retry loop on success
            else:
                # print(f"Failed to parse the result. Retrying...")
                time.sleep(1)  # Wait a bit before retrying
        else:
            fails[inst] = generated_text
            write_json(fails, fail_fname)
            print(f"Failed to process instruction after {max_try} attempts: {inst}")
    
    

def cache_noisy_language_goals(json_path, output_path, model_name):
    instructions = set()
    first_3_words = set()

    filtered_goals = 0
    language_goals = load_json(json_path)
    goal_keys = []
    for key, val in language_goals.items():
        goal_keys.append(key.lower())
        cleaned_instruction = clean_instruction(val.lower())

        if len(cleaned_instruction.split(" ")) > 55:
            filtered_goals += 1
            continue

        instructions.add(cleaned_instruction)
        first_3_words.add(" ".join(val.lower().split(" ")[:3]))

    print(
        "Total instructions: {}, Filtered: {}".format(
            len(instructions), filtered_goals
        )
    )
    max_instruction_len = 0
    for instruction in instructions:
        max_instruction_len = max(
            max_instruction_len, len(instruction.split(" "))
        )
    print("Max instruction length: {}".format(max_instruction_len))
    print("First 3 words: {}".format(first_3_words))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if model_name == "BERT":
        model = get_bert()
        text_embedding = model.batch_embed(list(instructions))
    else:
        model, _ = clip.load(model_name)
        batch = tokenize_and_batch(clip, list(instructions))

        with torch.no_grad():
            print(batch.shape)
            text_embedding = model.encode_text(batch.flatten(0, 1)).float()
    print(
        "Noisey Goals: {}, Embeddings: {}, Shape: {}".format(
            len(goal_keys), len(text_embedding), text_embedding[0].shape
        )
    )
    save_to_disk(text_embedding, goal_keys, output_path)


def cache_noisy_ovon_goals(json_path, output_path, model_name):
    categories = []

    filtered_goals = 0
    ovon_goals = load_json(json_path)
    goal_keys = []
    for key, val in ovon_goals.items():
        goal_keys.append(key.lower())
        val_selected = random.choice(val).lower()
        categories.append(val_selected)

    print(
        "Total categories: {}, Filtered: {}".format(
            len(categories), filtered_goals
        )
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    model, _ = clip.load(model_name)
    batch = tokenize_and_batch(clip, list(categories))

    with torch.no_grad():
        print(batch.shape)
        text_embedding = model.encode_text(batch.flatten(0, 1)).float()
    print(
        "Noisey OVON Goals: {}, Embeddings: {}, Shape: {}".format(
            len(goal_keys), len(text_embedding), text_embedding[0].shape
        )
    )
    save_to_disk(text_embedding, goal_keys, output_path)


def main(dataset_path, output_path, dataset, model, add_noise=False, split=None):
    if add_noise:
        if dataset == "lnav":
            cache_noisy_language_goals(dataset_path, output_path, model)
        else:
            cache_noisy_ovon_goals(dataset_path, output_path, model)
    elif dataset == "ovon":
        cache_ovon_goals(dataset_path, output_path)
    elif dataset == "lnav":
        cache_language_goals(dataset_path, output_path, model)
    elif dataset == "goat":
        cache_goat_goals(dataset_path, output_path, model, split)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="file path of OVON dataset",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="output path of clip features",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ovon",
        help="ovon",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="RN50",
    )
    parser.add_argument(
        "--noise",
        action="store_true",
        dest="noise",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=""
    )
    args = parser.parse_args()

    main(
        args.dataset_path,
        args.output_path,
        args.dataset,
        args.model,
        args.noise,
        args.split
    )
