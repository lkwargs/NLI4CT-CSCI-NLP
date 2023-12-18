import re
import transformers
import torch
from accelerate import Accelerator
from datasets import load_dataset
from trl import is_xpu_available
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import f1_score, precision_score, recall_score


def build_pipeline(checkpoint_path="output"):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True) 
    quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        # Copy the model to each device
    device_map = (
        {"": f"xpu:{Accelerator().local_process_index}"}
        if is_xpu_available()
        else {"": Accelerator().local_process_index}
    )
    torch_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=False,
        torch_dtype=torch_dtype,
        use_auth_token=True,
    )

    pipeline = transformers.pipeline(
        "text-generation",
        # model="meta-llama/Llama-2-7b-hf",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    return pipeline, tokenizer


def eval(pipeline: transformers.pipeline, tokenizer: AutoTokenizer):
    dataset = load_dataset('json', data_files={'dev': 'llama-2-dev.json'}, split='dev')

    results = []
    for dev_point in dataset:
        query, _, label = handle_datapoint(dev_point['text'])
        # query = "I am a dog. Please help me plan a surprise birthday party for my human, including fun activities, games and decorations. And don't forget to order a big bone-shaped cake for me to share with my fur friends!"
        # query = f"### Human: {query} ### Assistant: "
        best_answer = pipeline(
                        query,
                        do_sample=False,  # Set to False to get the most likely answer
                        # top_k=1,
                        num_return_sequences=1,  # Return only the best answer
                        eos_token_id=tokenizer.eos_token_id,
                        temperature=None,
                        max_new_tokens=128,
                    )
        print(query)
        print(best_answer)
        pred = handle_answer(best_answer[0]['generated_text'])
        print(pred)
        if pred is None:
            answers = pipeline(
                        query,
                        do_sample=True,  # Set to False to get the most likely answer
                        top_k=3,
                        num_return_sequences=1,  # Return only the best answer
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=512,
                    )
            for answer in answers:
                pred = handle_answer(answer['generated_text'])
                if pred:
                    break
            if label is None:
                label = "Entailment"
        results.append([pred, label])

    results_pred = []
    gold_labels = []
    for i in range(len(results)):
        if results[i][0] == "Entailment":
            results_pred.append(1)
        else:
            results_pred.append(0)
        if results[i][1] == "Entailment":
            gold_labels.append(1)
        else:
            gold_labels.append(0)

    f_score = f1_score(gold_labels, results_pred)
    p_score = precision_score(gold_labels, results_pred)
    r_score = recall_score(gold_labels, results_pred)

    print('F1:{:f}'.format(f_score))
    print('Precision Score:{:f}'.format(p_score))
    print('Recall Score:{:f}'.format(r_score))


DATA_PATTERN = re.compile(r"(<s>\[INST\] <<SYS>>.*?\[/INST\])(.*?)\"(.*?)\"(.*</s>)", re.DOTALL)

def handle_datapoint(datapoint: str):
    matchs = DATA_PATTERN.match(datapoint)
    query, answer, label, end = matchs.groups()
    answer += f"```{label}```{end}"
    return query, answer, label


def handle_answer(answer: str):
    if '"Entailment"' in answer:
        predicate = "Entailment"
    elif '"Contradiction"' in answer:
        predicate = "Contradiction"
    elif "contradict" in answer:
        predicate = "Contradiction"
    else:
        predicate = None

    return predicate


if __name__ == '__main__':
    # datapoint = '<s>[INST] <<SYS>>\nYou are a helpful assistant. You are going to determine the inference relation (entailment or contradiction) between pairs of Clinical Trial Reports (CTRs) and the statements, making claims about one of the summarized sections of the CTRs: Results.<</SYS>>. This task type is "Single". There is only one CTR. The statement is "there is a 13.2% difference between the results from the two the primary trial cohorts". The primary CTR section includes,\n Outcome Measurement: \n  Event-free Survival\n  Event free survival, the primary endpoint of this study, is defined as the time from randomization to the time of documented locoregional or distant recurrence, new primary breast cancer, or death from any cause.\n  Time frame: 5 years\nResults 1: \n  Arm/Group Title: Exemestane\n  Arm/Group Description: Patients receive oral exemestane (25 mg) once daily for 5 years.\n  exemestane: Given orally\n  Overall Number of Participants Analyzed: 3789\n  Measure Type: Number\n  Unit of Measure: percentage of participants  88        (87 to 89)\nResults 2: \n  Arm/Group Title: Anastrozole\n  Arm/Group Description: Patients receive oral anastrozole (1 mg) once daily for 5 years.\n  anastrozole: Given orally\n  Overall Number of Participants Analyzed: 3787\n  Measure Type: Number\n  Unit of Measure: percentage of participants  89        (88 to 90)[/INST] Based on the provided evidence, \n I think the relationship is "Contradiction". The statement "there is a 13.2% difference between the results from the two the primary trial cohorts" is likely to be false. The primary CTR section includes the results of the two primary trial cohorts, which show a difference of 88% and 89% in the event-free survival rate, respectively. This suggests that there is a significant difference between the two cohorts in terms of event-free survival, which contradicts the statement.'
    # print(handle_datapoint(datapoint))
    pipeline, tokenizer = build_pipeline("/home/josep/rl/nli4ct/slft/output")
    eval(pipeline, tokenizer)
