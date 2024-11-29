import json
import logging
import math
import os
import re
from pathlib import Path
from sys import excepthook

import numpy as np
import pandas as pd
from nltk import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

# Download required nltk resources (run once)
nltk.download('stopwords')
nltk.download('wordnet')

VALIDATION_FILES_PATH = Path("data/validation")
EXTRACTION_FILES_PATH = Path("data/validation/statistical_analysis")

os.makedirs(VALIDATION_FILES_PATH, exist_ok=True)
os.makedirs(EXTRACTION_FILES_PATH, exist_ok=True)

FRAME_ACTION_TYPES = [
    "Duty to document",
    "Duty to report",
    "Duty to designate",
    "Duty to inform a specific person",
    "Duty to disclose information",
    "Duty to provide access",
    "Duty to prove",
    "Duty to terminate",
    "Duty to facilitate",
    "Duty to assess",
    "Duty to consult",
    "Duty to investigate",
    "Duty to adopt measures"
]


def load_validation_files(target_path):
    file_dict = {}

    # Traverse the directory and its subdirectories
    for subfolder in os.listdir(target_path):
        subfolder_path = os.path.join(target_path, subfolder)

        if os.path.isdir(subfolder_path):  # Check if it's a directory
            for file_name in os.listdir(subfolder_path):
                if file_name.endswith('.txt'):  # Only process .txt files
                    file_path = os.path.join(subfolder_path, file_name)

                    # Open and read the file content
                    with open(file_path, 'r') as file:
                        content = file.read()

                    # Add the file name and content to the dictionary
                    file_dict[file_name] = content

    return file_dict


def extract_evaluation_from_file(content):
    start_string = "**Grades**"
    end_string = "## LLM OUTPUT"

    start_pos = content.find(start_string)
    end_pos = content.find(end_string, start_pos)

    extracted_content = content[start_pos:end_pos].strip()
    return extract_grades(extracted_content)


def extract_grades(content):
    # Regular expression to capture the classification name and value (numbers)
    pattern = r"(\w[\w\s-]*\w):\s*\[(\d+)\]"

    # Find all matches in the content
    matches = re.findall(pattern, content)

    # Convert the matches into a dictionary
    grades_dict = {match[0].strip(): int(match[1]) for match in matches}

    return grades_dict


def extract_json_from_file(file_content):
    # Use regular expression to find JSON blocks

    file_content = file_content.split("## LLM OUTPUT")[1]
    file_content = file_content.split("```")[1]

    json_data = json.loads(file_content)
    return json_data


def extract_metadata_from_file(file_content):
    # Extract metadata using regex
    metadata_pattern = r"## Metadata\s+Article:\s*(\S+)\s+Paragraph:\s*(\S+)\s+LLM Model:\s*(\S+)\s+Prompt:\s*(\S+)\s+Timestamp:\s*(\S+)"
    metadata_match = re.search(metadata_pattern, file_content)

    # If matches found, organize them into a dictionary
    if metadata_match:
        return {
            "Article": metadata_match.group(1),
            "Paragraph": metadata_match.group(2),
            "LLM_Model": metadata_match.group(3),
            "Prompt": metadata_match.group(4),
            "Timestamp": metadata_match.group(5)
        }
    else:
        return None


def extract_context_from_file(file_content):
    # Extract full provision
    provision_pattern = r"### Full provision \(For reference\)\s*(.*?)(?=\n## Sentence)"
    provision_match = re.search(provision_pattern, file_content, re.DOTALL)

    # If matches found, organize them into a dictionary
    if provision_match:
        return provision_match.group(1).strip()
    else:
        return None


def extract_sentence_from_file(file_content: str) -> str:
    # Extract the sentence using regex
    sentence_pattern = r"## Sentence \(Processed by the LLM\):\s*(.*?)(?=\n## Evaluation:|$)"
    sentence_match = re.search(sentence_pattern, file_content, re.DOTALL)

    if sentence_match:
        return sentence_match.group(1).strip()
    else:
        return None


def calculate_statistics():
    validation_files = load_validation_files(EXTRACTION_FILES_PATH)

    obligation_types = {
        "Obligation of Being": {
            "total": 0,
            "HasBE": 0
        },
        "Obligation of Action": {
            "total": 0,
            "HasBE": 0
        }
    }

    most_burdened_persons = {
        "Not stated": 0
    }

    action_type_classification = {
        "Invented Action Type": 0
    }

    invented_action_types = {}

    beneficiaries = {
        "Not stated": 0
    }

    def process_text(target):
        # Initialize lemmatizer and stopwords
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        # Tokenize and lemmatize each word, removing stopwords
        words = target.lower().split()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        normalized = " ".join(words)

        return normalized

    def process_obligation_type(target):
        for obligation_object in target:
            if "ObligationTypeClassification" not in obligation_object.keys():
                print("No obligation type classification")
                continue
            extracted_obligation_type = obligation_object["ObligationTypeClassification"]
            if extracted_obligation_type in obligation_types:
                obligation_types[extracted_obligation_type]["total"] += 1
            else:
                print("Unkown obligation type", extracted_obligation_type)
            burdened_entities = obligation_object["BurdenedPersons"]

            for burdened_entity in burdened_entities:
                if burdened_entity["stated"] == "yes":
                    obligation_types[extracted_obligation_type]["HasBE"] += 1
                    break
            if extracted_obligation_type == "Obligation of Being" and obligation_types[extracted_obligation_type][
                "HasBE"] == 1:
                print("has BE", obligation_object)

    def process_beneficiaries(target):
        for obligation_object in target:

            if "Beneficiaries" not in obligation_object.keys():
                continue

            beneficiaries_entities = obligation_object["Beneficiaries"]

            for beneficiary_entity in beneficiaries_entities:
                if beneficiary_entity["stated"] == "yes":
                    beneficiary_entity_value = process_text(beneficiary_entity["value"])
                    if beneficiary_entity_value not in beneficiaries:
                        beneficiaries[beneficiary_entity_value] = 1
                    else:
                        beneficiaries[beneficiary_entity_value] += 1
                else:
                    beneficiaries["Not stated"] += 1

    def process_burdened_persons(target):

        replacements = {
            "deployers ai system generates manipulates image, audio video content constituting deep fake": "deployers of AI system",
            "deployers ai system generates manipulates text published purpose informing public matter public interest": "deployers of AI system",
            "member state concerned": "member state",
            "deployers": "deployer",
            "provider general-purpose ai model": "provider general-purpose ai model",
            "provider general-purpose ai model adhere approved code practice comply european harmonised standard": "provider general-purpose ai model",
            "provider general-purpose ai model concerned": "provider general-purpose ai model",
            "provider general-purpose ai model concerned, representative": "provider general-purpose ai model",
            "provider general-purpose ai model placed market 2 august 2025": "provider general-purpose ai model",
            "provider general-purpose ai model systemic risk": "provider general-purpose ai model",
        }
        for obligation_object in target:

            if "BurdenedPersons" not in obligation_object.keys():
                continue
            burdened_entities = obligation_object["BurdenedPersons"]

            for burdened_entity in burdened_entities:

                if burdened_entity["stated"] == "yes":
                    burdened_entity_value = process_text(burdened_entity["value"])

                    if burdened_entity_value in replacements:
                        burdened_entity_value = replacements[burdened_entity_value]

                    if burdened_entity_value not in most_burdened_persons:
                        most_burdened_persons[burdened_entity_value] = 1
                    else:
                        most_burdened_persons[burdened_entity_value] += 1
                else:
                    most_burdened_persons["Not stated"] += 1

    def process_action_types(target):

        for action_object in target:
            if "ActionTypeClassification" not in action_object.keys():

                if "NoAction" not in action_type_classification:
                    action_type_classification["NoAction"] = 1
                else:
                    action_type_classification["NoAction"] += 1

                continue
            extracted_action_types = action_object["ActionTypeClassification"]
            extracted_action_types = [s.strip() for s in extracted_action_types.split("/")]

            for extracted_action_type in extracted_action_types:

                if extracted_action_type in FRAME_ACTION_TYPES:

                    if extracted_action_type in action_type_classification:
                        action_type_classification[extracted_action_type] += 1
                    else:
                        action_type_classification[extracted_action_type] = 1
                else:
                    print(f"Invented: {extracted_action_type}")

                    if extracted_action_type in invented_action_types:
                        invented_action_types[extracted_action_type] += 1
                    else:
                        invented_action_types[extracted_action_type] = 1
                    action_type_classification["Invented Action Type"] += 1

    for file in validation_files:
        content = validation_files[file]
        json_content = extract_json_from_file(content)

        process_obligation_type(json_content)
        process_burdened_persons(json_content)
        process_action_types(json_content)
        process_beneficiaries(json_content)

    action_type_classification = dict(
        sorted(action_type_classification.items(), key=lambda item: item[1], reverse=True))
    most_burdened_persons = dict(
        sorted(most_burdened_persons.items(), key=lambda item: item[1], reverse=True))

    # Store the values
    print(json.dumps(obligation_types, indent=2))
    print(json.dumps(most_burdened_persons, indent=2))
    print(json.dumps(action_type_classification, indent=2))

    store_json_as_excel(target_dict=action_type_classification,
                        target_path=Path(EXTRACTION_FILES_PATH) / "action_type.xlsx",
                        columns=["Action Type", "Count"])

    store_json_as_excel(target_dict=most_burdened_persons,
                        target_path=Path(EXTRACTION_FILES_PATH) / "addresses.xlsx",
                        columns=["Addresses", "Count"])

    store_obligation_type(obligation_types)

    store_json_as_excel(target_dict=beneficiaries,
                        target_path=Path(EXTRACTION_FILES_PATH) / "beneficiaries.xlsx",
                        columns=["Beneficiaries", "Count"])

    store_json_as_excel(target_dict=invented_action_types,
                        target_path=Path(EXTRACTION_FILES_PATH) / "invented_actionx.xlsx",
                        columns=["Action", "Count"]
                        )


def store_obligation_type(target):
    # Convert JSON to DataFrame
    df = pd.DataFrame(target).T.reset_index()
    df.columns = ["Obligation Type", "Total", "HasBE"]

    # Save DataFrame to Excel
    output_path = EXTRACTION_FILES_PATH / "ObligationData.xlsx"
    df.to_excel(output_path, index=False)

    print(f"Data saved to {output_path}")


def store_json_as_excel(target_dict, target_path, columns):
    df = pd.DataFrame(list(target_dict.items()), columns=columns)
    df.to_excel(target_path, index=False)


def calculate_validation_metrics():
    # load files
    validation_files = load_validation_files(VALIDATION_FILES_PATH)

    validation_evaluations = []

    for validation_file in validation_files:
        print("=====")
        result = extract_evaluation_from_file(validation_files[validation_file])

        result["file_name"] = validation_file

        validation_evaluations.append(result)

    print(json.dumps(validation_evaluations, indent=4))

    # Convert the list of dictionaries into a Pandas DataFrame
    df = pd.DataFrame(validation_evaluations)

    # Reorder columns to make 'file_name' the first column
    cols = ['file_name'] + [col for col in df.columns if col != 'file_name']
    df = df[cols]

    df.to_excel("data/regulations/validation/results.xlsx", index=False)

    accuracies = calculate_accuracy_and_mean(df)
    print(json.dumps(accuracies, indent=4))
    acc = flatten_results(accuracies)


def flatten_results(data):
    # Flatten the JSON structure into a list of dictionaries
    flattened_data = []
    for entity, values in data.items():
        flattened_data.append({
            "Entity": entity,
            "Accuracy": values["Accuracy"],
            "Mean": values["Mean"]
        })

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(flattened_data)

    # Save the DataFrame to CSV
    df.to_excel('data/regulations/validation/acc_mean.xlsx', index=False)

    # Display the DataFrame
    print(df)


def calculate_accuracy_and_mean(df):
    results = {}
    for column in df.columns:
        if column == 'file_name':
            continue
        mean = float(df[column].mean(skipna=True))
        if column.lower().find("classification") >= 0:  # For classification columns
            accuracy = mean
        elif column != 'file_name':  # For other columns, 2 is correct

            values = list(df[column])
            values = [float(v) for v in values if v is not None and not math.isnan(v)]
            values = [1 if v == 2 else 0 for v in values]
            accuracy = float(np.sum(values) / len(values))
        else:
            accuracy = 0
        # Mean calculation
        mean = float(df[column].mean(skipna=True))

        # Store accuracy and mean for the column
        results[column] = {'Accuracy': accuracy, 'Mean': mean}

    return results


def extract_all_as_dict():
    validation_files = load_validation_files(VALIDATION_FILES_PATH)

    data = []

    for file in validation_files:
        content = validation_files[file]
        try:
            json_content = extract_json_from_file(content)
        except Exception as e:
            print(f"Error while processing json: {e}")
            print("Content: {}".format(content))
        metadata = extract_metadata_from_file(content)
        full_provision = extract_context_from_file(content)
        sentence = extract_sentence_from_file(content)

        if metadata is None:
            metadata = {}
            print(f"Metadata not found for file: {content}")
        metadata["full_provision"] = full_provision
        metadata["sentence"] = sentence
        metadata["json_content"] = json_content

        data.append(metadata)

    return data


def export_beneficiaries_as_list(target_deontic_structure: list, target_info) -> str:
    def _process_single_beneficiary(beneficiary):
        value = beneficiary["value"]
        stated = beneficiary["stated"]
        return f"{value} (stated={stated})"

    full_provision = target_deontic_structure["full_provision"]  if "full_provision" in target_deontic_structure else None
    sentence = target_deontic_structure["sentence"] if "sentence" in target_deontic_structure else None
    article = target_deontic_structure["Article"] if "Article" in target_deontic_structure else None
    paragraph = target_deontic_structure["Paragraph"] if "Paragraph" in target_deontic_structure else None
    template_string = \
        f"""
## Article (Paragraph):
{article} ({paragraph})
## Full provision (Context):
{full_provision}
## Sentence (for LLM):       
{sentence}"""
    for index, deontic_relation in enumerate(target_deontic_structure["json_content"]):
        content_target_info = []
        if target_info in deontic_relation:
            content_target_info = [_process_single_beneficiary(b)
                             for b in deontic_relation[target_info]]

        template_string += \
            f"""
### Obligation {index + 1}
{target_info}:           
"""

        for b in content_target_info:
            template_string += f"-> {b}"

    if not (article and paragraph):
        return ""

    return template_string

