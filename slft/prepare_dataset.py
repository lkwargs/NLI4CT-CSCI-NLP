import os
import json
import numpy
import subprocess


def generate_cot_dataset(json_path="../training_data/train.json", output_path="llama-2-train.json"):
    data_zip_path = "../training_data.zip"
    data_path = "../training_data"
    if not os.path.exists(data_path):
        unzip_cmd = f"unzip {data_zip_path}"
        subprocess.run(unzip_cmd, shell=True)

    CT_json_dir = os.path.join(data_path, "CT json")
    with open(json_path) as json_file:
        subdata_set = json.load(json_file)

    subdata_uuid_list = list(subdata_set.keys())
    subdata_statements = [subdata_set[subdata_uuid_list[i]]["Statement"] for i in range(len(subdata_uuid_list))]

    subdata_json_list = []
    for i in range(len(subdata_uuid_list)):
        datapoint = subdata_set[subdata_uuid_list[i]]

        primary_ctr_path = os.path.join(CT_json_dir, datapoint["Primary_id"]+".json")
        with open(primary_ctr_path) as json_file:
            primary_ctr = json.load(json_file)
        
        sys_prompt = f"You are a helpful assistant. You are going to determine the inference relation (entailment or contradiction) between pairs of Clinical Trial Reports (CTRs) and the statements, making claims about one of the summarized sections of the CTRs: {datapoint['Section_id']}."
        input_prompt = [f"This task type is \"{datapoint['Type']}\"."]
        if datapoint["Type"] == "Comparison":
            input_prompt += ["There are multiple CTRs."]
        else:
            input_prompt += ["There is only one CTR."]

        #Retrieve the full section from the primary trial
        primary_section = primary_ctr[datapoint["Section_id"]]

        statement = subdata_statements[i]
        input_prompt += [f"The statement is \"{statement}\"."]
        input_prompt += [f"The primary CTR section includes,\n"]
        input_prompt += ['\n'.join(primary_section)]
        #Repeat for the secondary trial
        if datapoint["Type"] == "Comparison":
            secondary_ctr_path = os.path.join(CT_json_dir, datapoint["Secondary_id"]+".json")
            with open(secondary_ctr_path) as json_file:
                secondary_ctr = json.load(json_file)
            secondary_section = secondary_ctr[datapoint["Section_id"]]

            input_prompt += [f"The secondary CTR section includes,\n"]
            input_prompt += ['\n'.join(secondary_section)]

        output_prompt = ["Based on the provided evidence, \n"]
        # output_prompt += ["I found the evidence in those lines are important, \nPrimary_evidence_index: " + str(datapoint["Primary_evidence_index"])]
        # if datapoint["Type"] == "Comparison":
        #     output_prompt += ["Secondary_evidence_index: " + str(datapoint["Secondary_evidence_index"])]

        output_prompt += [f"I think the relationship is \"{datapoint['Label']}\"."]
        input_prompt, output_prompt = ' '.join(input_prompt), ' '.join(output_prompt)
        subdata_json_list.append({
            "text": f"<s>[INST] <<SYS>>\n{sys_prompt}<</SYS>>. {input_prompt}[/INST] {output_prompt}</s>"
        })

    print(subdata_json_list)
    with open(output_path, 'w') as jsonFile:
        jsonFile.write(json.dumps(subdata_json_list, indent=4))


if __name__ == '__main__':
    generate_cot_dataset(json_path="../training_data/train.json", output_path="llama-2-train.json")
    generate_cot_dataset(json_path="../training_data/dev.json", output_path="llama-2-dev.json")
