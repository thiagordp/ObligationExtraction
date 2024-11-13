import time
from datetime import datetime, timedelta
from pathlib import Path
import json

from typing import Dict, List, Any, Tuple

import numpy as np
import tqdm
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer, util

import seaborn as sns

import shutil
import os
import random
import glob

import logging
import tiktoken

from src.llm import GroqApiClient, run_prompt

OBLIGATIONS_DATA = "data/raw/ai_act_provisions.json"
SELECTED_PROVISIONS = "data/raw/selected_provisions.json"
IGNORED_PROVISIONS = "data/raw/ignored_provisions.json"
FILTERED_JSON_DATA = "data/raw/obligationsSubjects.json"
LLM_MODEL = "llama-3.1-70b-versatile"

SYSTEM_PROMPT = "data/prompts/system_prompt.txt"
USER_PROMPT = "data/prompts/user_prompt.txt"
VALIDATION_SAMPLE_SIZE = 40


def setup_logging(level=logging.INFO):
    # Get current timestamp for log file name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"logs/log_{timestamp}.log"

    # Define the log format to include timestamp, file, function, and message
    log_format = "%(asctime)s - %(levelname)s\t-\t%(filename)s\t-\t%(funcName)s\t-\t%(message)s"

    # Create a logger object
    logger = logging.getLogger()
    logger.setLevel(level)  # Set the minimum logging level

    # Create a file handler to write log to a file
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S"))

    # Create a console handler to output log to the screen
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S"))

    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Example to log a test message
    logger.info("Logging is set up. Logs will be saved to console and file.")


def load_json(target: Path) -> dict:
    # Open and read the JSON file
    with open(target, 'r') as file:
        data = json.load(file)

    return data


def store_json(target: Path, data: dict | list[Dict]) -> None:
    with open(target, 'w') as file:
        json.dump(data, file, indent=4)


def add_article_paragraph(target: dict, article, paragraph, content):
    if article not in target:
        target[article] = {}
    target[article][paragraph] = content

    return target


class EmbeddingService:
    """Service to handle embedding and similarity calculations."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encodes a list of texts into embeddings."""
        return self.model.encode(texts)

    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Computes a cosine similarity matrix for given embeddings."""
        return util.cos_sim(embeddings, embeddings).numpy()


class SampleSelector:
    """Selects k samples with the least similarity among them and provides an option to plot similarity matrix."""

    def __init__(self, embedding_service):
        self.embedding_service = embedding_service

    def select_least_similar_samples(self, samples: Dict[str, str], k: int) -> Dict[str, str]:
        """Selects k samples with minimal similarity to each other."""

        # Filtering too long paragraphs.

        sample_keys = list(samples.keys())
        sample_texts = list(samples.values())

        # Step 1: Generate embeddings
        embeddings = self.embedding_service.encode_texts(sample_texts)

        # Step 2: Calculate similarity matrix
        similarity_matrix = self.embedding_service.compute_similarity_matrix(embeddings)

        # Step 3: Find the sample closest to the median similarity sum
        similarity_sums = np.mean(similarity_matrix, axis=1)
        median_similarity = np.median(similarity_sums)
        median_diffs = np.abs(similarity_sums - median_similarity)
        selected_indices = [np.argmin(median_diffs)]

        logging.info(f"Median similar sample. Index: {selected_indices[0]}, Key: {sample_keys[selected_indices[0]]}")

        # Step 4: Iteratively select samples with least similarity to already selected
        while len(selected_indices) < k:
            remaining_indices = [i for i in range(len(samples)) if i not in selected_indices]
            similarity_to_selected = similarity_matrix[remaining_indices][:, selected_indices]
            mean_similarity = np.mean(similarity_to_selected, axis=1)
            next_index = remaining_indices[np.argmin(mean_similarity)]
            selected_indices.append(next_index)

        # Return selected samples
        selected_sample_keys = [sample_keys[i] for i in selected_indices]
        return {key: samples[key] for key in selected_sample_keys}

    def plot_similarity_matrix(self, samples: Dict[str, str]):
        """Plots the similarity matrix as a black-and-white heatmap with sample keys as labels."""
        sample_keys = list(samples.keys())
        sample_texts = list(samples.values())

        # Generate embeddings and similarity matrix
        embeddings = self.embedding_service.encode_texts(sample_texts)
        similarity_matrix = self.embedding_service.compute_similarity_matrix(embeddings)

        # Plot heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(similarity_matrix, annot=False, cmap="Greys", cbar=True)
        plt.title("Similarity Matrix (Black and White)")
        plt.xlabel("Samples")
        plt.ylabel("Samples")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()


def remove_sentence_subject(context, sentence, subject):
    entities_to_ignore = ["Regulation", "Section", "Article", "Paragraph", "Subparagraph", "Procedure", "Derogation"]

    for entity in entities_to_ignore:
        if subject.lower().find(entity.lower()) >= 0:
            return True

    return False


def plot_token_distribution(data: List[int], output_path: Path):
    """
    Plot the histogram of token counts and save it as a PDF.

    Args:
        data (List[int]): The list of token counts.
        output_path (Path): The path to save the plot.
    """
    plt.figure(figsize=(10, 6))

    # Calculate quartiles
    Q1, Q2, Q3 = np.percentile(data, [25, 50, 75])

    # Plot histogram
    plt.hist(data, bins=30, color='lightblue', edgecolor='black', alpha=0.7)

    # Add lines for each quartile
    plt.axvline(Q1, color='purple', linestyle='--', linewidth=1.5, label='Q1 (25th percentile)')
    plt.axvline(Q2, color='red', linestyle='--', linewidth=1.5, label='Median (50th percentile)')
    plt.axvline(Q3, color='green', linestyle='--', linewidth=1.5, label='Q3 (75th percentile)')

    # Add legend and labels
    plt.legend()
    plt.xlabel('Tokens')
    plt.ylabel('Frequency')
    plt.title('Histogram of Token Counts')

    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def process_provision_data(target: Dict) -> Tuple[str, str, List]:
    """
    Process the provision data to extract relevant details.

    Args:
        target (Dict): The provision data.

    Returns:
        Tuple[str, str, List]: The provision ID, provision text, and list of sentences.
    """
    provision_id = target.get("par_id", "")
    provision_text = target.get("text", "")
    sentences = target.get("sentences", [])

    return provision_id, provision_text, sentences


def filter_sentence(provision_text: str, sentence_content: str, sentence_subject: Any, sentence_tokens: int) -> bool:
    """
    Determine if a sentence should be included or ignored.

    Args:
        provision_text (str): The full provision text.
        sentence_content (str): The sentence text.
        sentence_subject (Any): The subject of the sentence.
        sentence_tokens (int): The number of tokens in the sentence.

    Returns:
        bool: True if the sentence should be ignored, False if it should be included.
    """
    if isinstance(sentence_subject, list) or sentence_tokens > 200:
        return True
    if remove_sentence_subject(provision_text, sentence_content, sentence_subject):
        return True
    return False


def process_sentences(provision_id: str, provision_text: str, sentences: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Process sentences of a provision and categorize them as included or ignored.

    Args:
        provision_id (str): The provision ID.
        provision_text (str): The full provision text.
        sentences (List[Dict]): The list of sentences.

    Returns:
        Tuple[List[Dict], List[Dict]]: The list of included and ignored sentences.
    """
    included_sentences = []
    ignored_sentences = []

    for sentence_index, sentence in enumerate(sentences):
        sentence_content = sentence.get("text", "")
        sentence_subject = sentence.get("subject", "")
        sentence_tokens = count_tokens(sentence_content)
        sentence_provision_proportion = float(len(sentence_content) / len(provision_text)) if provision_text else 0.0

        prov_sentence = {
            "provision_id": provision_id,
            "sentence_id": sentence_index,
            "full_text": provision_text,
            "sentence": sentence_content,
            "subject": sentence_subject,
            "sentence_provision_proportion": sentence_provision_proportion,
            "sentence_tokens": sentence_tokens,
        }

        if filter_sentence(provision_text, sentence_content, sentence_subject, sentence_tokens):
            ignored_sentences.append(prov_sentence)
        else:
            included_sentences.append(prov_sentence)

    return included_sentences, ignored_sentences


def organize_filtered_data(obligation_path):
    """
    Process provisions by filtering sentences based on certain criteria and store the results.

    Returns:
        Tuple[List[Dict], List[Dict]]: The included and ignored provisions.
    """

    if type(obligation_path) != Path:
        obligation_path = Path(str(obligation_path))

    provisions = load_json(obligation_path)

    all_included_provisions = []
    all_ignored_provisions = []

    logging.info("Processing provisions...")

    for index, provision in tqdm.tqdm(enumerate(provisions), total=len(provisions)):
        logging.info(f"Processing provision {index + 1}/{len(provisions)}")
        provision_id, provision_text, sentences = process_provision_data(provision)

        included_sentences, ignored_sentences = process_sentences(provision_id, provision_text, sentences)

        all_included_provisions.extend(included_sentences)
        all_ignored_provisions.extend(ignored_sentences)

    logging.info("Finished processing provisions.")

    logging.info(f"Ignored sentences: {len(all_ignored_provisions)}")
    logging.info(f"Included sentences: {len(all_included_provisions)}")

    store_json(Path(SELECTED_PROVISIONS), all_included_provisions)
    store_json(Path(IGNORED_PROVISIONS), all_ignored_provisions)

    token_counts = [v["sentence_tokens"] for v in all_included_provisions]

    plot_token_distribution(token_counts, Path("data/outputs/token_distribution_sentences.pdf"))

    return all_included_provisions, all_ignored_provisions


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Counts the number of tokens in the given text for a specific model.

    Parameters:
    - text (str): The input string to tokenize.
    - model (str): The model for which to calculate tokens, default is "gpt-3.5-turbo".

    Returns:
    - int: The number of tokens in the text.
    """
    # Initialize encoding for the specific model
    encoding = tiktoken.encoding_for_model(model)

    # Encode the text to get tokens
    tokens = encoding.encode(text)

    # Return the count of tokens
    return len(tokens)


def run_llm(sentence: str, context: str, model_name: str):
    client = GroqApiClient().get_client()

    system_prompt = open(SYSTEM_PROMPT).read()
    user_prompt = open(USER_PROMPT).read()

    user_prompt = user_prompt.replace("@SENTENCE", sentence)
    user_prompt = user_prompt.replace("@CONTEXT", context)

    result = run_prompt(
        client=client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_name=model_name
    )

    input_tokens = "\n".join([system_prompt, user_prompt])
    output_tokens = str(result)

    return result, count_tokens(input_tokens), count_tokens(output_tokens)


def divide_sample():
    file_paths = glob.glob("data/outputs/*.txt")

    # Define destination folders for each expert
    expert_folders = {
        1: "data/outputs/Expert1",
        2: "data/outputs/Expert2",
        3: "data/outputs/Expert3"
    }

    # Ensure the folders for each expert exist
    for folder in expert_folders.values():
        os.makedirs(folder, exist_ok=True)

    # Initialize lists to keep track of file assignments
    expert1_files = []
    expert2_files = []
    expert3_files = []

    # Shuffling files list.
    random.shuffle(file_paths)

    file_paths = file_paths[:VALIDATION_SAMPLE_SIZE]

    # Balanced assignment of files to experts
    for index, file in enumerate(file_paths):
        if index % 3 == 0:
            expert1_files.append(file)
            # expert2_files.append(file)
        elif index % 3 == 1:
            expert2_files.append(file)
            # expert3_files.append(file)
        else:  # index % 3 == 2
            expert3_files.append(file)
            # expert1_files.append(file)

    # random.seed(10)
    random.shuffle(expert1_files)
    random.shuffle(expert2_files)
    random.shuffle(expert3_files)

    # Move files to the corresponding expert folders
    for file in expert1_files:
        shutil.copy(file, os.path.join(expert_folders[1], os.path.basename(file)))

    for file in expert2_files:
        shutil.copy(file, os.path.join(expert_folders[2], os.path.basename(file)))

    for file in expert3_files:
        shutil.copy(file, os.path.join(expert_folders[3], os.path.basename(file)))

    logging.info("Files have been moved to expert folders.")


def create_labeling_template(samples: dict, target_key: str, llm_output: dict, input_token_count: int,
                             output_token_count: int) -> None:
    with open("data/raw/sample_template.txt") as fp:
        template_text = fp.read()

    article, paragraph = target_key.split(".")

    template_text = template_text.replace("@ARTICLE", article.strip())
    template_text = template_text.replace("@PARAGRAPH", paragraph.strip())
    template_text = template_text.replace("@TEXT", samples["full_text"].strip())
    template_text = template_text.replace("@SENTENCE", samples["sentence"].strip())
    template_text = template_text.replace("@LLM_OUTPUT", json.dumps(llm_output, indent=3))
    template_text = template_text.replace("@LLM_MODEL", LLM_MODEL)
    template_text = template_text.replace("@INPUT_TOKENS", str(input_token_count))
    template_text = template_text.replace("@OUTPUT_TOKENS", str(output_token_count))
    template_text = template_text.replace("@PROMPT", str(SYSTEM_PROMPT))
    template_text = template_text.replace("@TIMESTAMP", datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])

    with open(f"data/outputs/{target_key}.txt", "w") as fp:
        fp.write(template_text)


def calculate_expected_finish(start_time, current_iteration, total_iterations):
    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Calculate average time per iteration
    avg_time_per_iteration = elapsed_time / current_iteration

    # Calculate remaining time and expected finish time
    remaining_iterations = total_iterations - current_iteration
    expected_finish_time = datetime.now() + timedelta(seconds=remaining_iterations * avg_time_per_iteration)

    # Return the expected finish time
    return expected_finish_time


def obligation_extraction():
    def calculate_filename(s):
        result = f"{s['provision_id']}_{s['sentence_id']}.txt"
        return result

    # Create Client
    logging.info("Processing dataset")

    samples = load_json(Path(SELECTED_PROVISIONS))
    # random.seed(10)
    random.shuffle(samples)

    processed_files = glob.glob("data/outputs/*.txt")
    processed_files = [s.replace("data/outputs/", "") for s in processed_files]

    # Avoid processing files already processed.
    samples = [s for s in samples if calculate_filename(s) not in processed_files]

    start_time = time.time()
    for index, sample in tqdm.tqdm(enumerate(samples)):
        logging.info(f"Processed {index + 1} out of {len(samples)}")

        expected_finish_time = calculate_expected_finish(start_time, index + 1, len(samples))
        logging.info(f"Expected finish time is {expected_finish_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Run LLM call
        output, in_token, out_token = run_llm(sentence=sample["sentence"], context=sample["full_text"],
                                              model_name=LLM_MODEL)
        # Put the LLM structure
        target_key = sample["provision_id"] + "_" + str(sample["sentence_id"])
        create_labeling_template(samples=sample, target_key=target_key, llm_output=output,
                                 input_token_count=in_token,
                                 output_token_count=out_token)
