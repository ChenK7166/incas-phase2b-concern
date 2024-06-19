from transformers import AutoTokenizer
import transformers
import pandas as pd
from tqdm import tqdm
import argparse
import warnings
import json
import logging
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

CONCERN_DICT = {
    "us": "Concern/US Military",
    "defense": "Concern/Defense",
    "sovereignty": "Concern/Territorial Sovereignty",
    "economy": "Concern/Domestic Economy",
    "trade": "Concern/Trade",
    "separatism": "Concern/Insurgency Separatism",
    "crime": "Concern/Crime",
    "noneother": "Concern/NoneOther"
}

MATCH_CONCERN_LIST = CONCERN_DICT.keys()
FORMAL_CONCERN_LIST = CONCERN_DICT.values()

def truncate_sentence(sentence, max_tokens=256):
    tokens = word_tokenize(sentence)

    if len(tokens) <= max_tokens:
        return sentence  # No truncation needed

    truncated_tokens = tokens[:max_tokens]
    truncated_sentence = ' '.join(truncated_tokens)
    truncated_sentence = truncated_sentence + " ..."

    return truncated_sentence

# Load prompt template
prompt_file_path = 'prompt_inference.txt'
with open(prompt_file_path, 'r') as file:
    prompt_template = file.read()

def build_prompt(system_prompt, user_message):
    if system_prompt is not None:
        SYS = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>"
    else:
        SYS = ""
    CONVO = ""
    SYS = "<s>" + SYS
    CONVO += f"[INST] {user_message} [/INST]"
    return SYS + CONVO

def label_concern(input_list):

    # concate prompt with tweets
    prompt_list = []
    sys_prompt = prompt_template
    for user_message in input_list:
        user_message = truncate_sentence(user_message)
        prompt = build_prompt(sys_prompt, "tweet: "+user_message)
        prompt_list.append(prompt)

    print("loading model and tokenizer ...")
    model = "incas_tuned_model_2b/incas_tuned_model_2b_v1/"
    tokenizer = AutoTokenizer.from_pretrained(model, max_length=512)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        device_map="cpu",
        max_new_tokens=100,
        return_full_text=False
    )

    print("inference result ...")
    predict_results = []
    for i in tqdm(range(len(prompt_list))):
        result = pipeline(prompt_list[i])
        # predict_raw_results.append()
        raw_output = result[0]["generated_text"][len(prompt_list[i]):]
        output_result = {key: 0 for key in FORMAL_CONCERN_LIST}
        raw_output = raw_output.lower()
        is_match = 0
        for concern in MATCH_CONCERN_LIST:
            if concern in raw_output:
                output_result[CONCERN_DICT[concern]] = 1
                is_match += 1
        if is_match == 0:
            output_result["Concern/NoneOther"] = 1
        predict_results.append(output_result)

    return predict_results

if __name__ == "__main__":
    """
        Args:
            messages: list of dictionaries [{'id':xxx,'contentText':xxx,...},...]

        Returns:
            A dict whose keys are message IDs and values are lists of annotations.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="small_sample.jsonl")
    args = parser.parse_args()

    # only process Twitter data
    print("loading json file ...")
    df = pd.read_json(args.file, lines=True)
    # messages = [json.loads(line) for line in file]

    contentText_list = df["contentText"].tolist()
    id_list = df["id"].tolist()

    label_list = label_concern(contentText_list)

    print("annotating messages ...")
    annotation_list = []
    for i in range(len(id_list)):
        annotations = {}
        annotation = {}
        annotation["id"] = id_list[i]
        annotation["contentText"] = contentText_list[i]
        annotation["concern"] = label_list[i]
        annotation["providerName"] = "ta1-usc-isi"
        annotations[id_list[i]] = [annotation]
        annotation_list.append(annotations)

    with open('concern_annotate.jsonl', 'w') as json_file:
        for annotation in annotation_list:
            json_file.write(json.dumps(annotation) + '\n')


