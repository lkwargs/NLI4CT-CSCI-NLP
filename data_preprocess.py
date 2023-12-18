import os
import json
import numpy
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import f1_score,precision_score,recall_score


data_zip_path = "training_data.zip"
data_path = "training_data"
if not os.path.exists(data_path):
    unzip_cmd = f"unzip {data_zip_path}"
    subprocess.run(unzip_cmd, shell=True)


dev_path = "training_data/dev.json"
CT_json_dir = os.path.join(data_path, "CT json")
with open(dev_path) as json_file:
    dev = json.load(json_file)

# Example instance
print(dev[list(dev.keys())[1]])

uuid_list = list(dev.keys())
statements = []
gold_dev_primary_evidence = []
gold_dev_secondary_evidence = []
for i in range(len(uuid_list)):
    #Retrieve all statements from the development set
    statements.append(dev[uuid_list[i]]["Statement"])

Results = {}

for i in range(len(uuid_list)):
    primary_ctr_path = os.path.join(CT_json_dir, dev[uuid_list[i]]["Primary_id"]+".json")
    with open(primary_ctr_path) as json_file:
        primary_ctr = json.load(json_file)
    
    #Retrieve the full section from the primary trial
    primary_section = primary_ctr[dev[uuid_list[i]]["Section_id"]]

    #Convert a primary section entries to a matrix of TF-IDF features.
    vectorizer = TfidfVectorizer().fit(primary_section)
    X_s = vectorizer.transform([statements[i]])
    X_p = vectorizer.transform(primary_section)
    #Compute the cosine similarity between the primary section entries and the statement
    primary_scores = cosine_distances(X_s, X_p)
    #Repeat for the secondary trial
    if dev[uuid_list[i]]["Type"] == "Comparison":
        secondary_ctr_path = os.path.join(CT_json_dir, dev[uuid_list[i]]["Secondary_id"]+".json")
        with open(secondary_ctr_path) as json_file:
            secondary_ctr = json.load(json_file)
        secondary_section = secondary_ctr[dev[uuid_list[i]]["Section_id"]]
        vectorizer = TfidfVectorizer().fit(secondary_section)
        X_s = vectorizer.transform([statements[i]])
        X_p = vectorizer.transform(secondary_section)
        secondary_scores = cosine_distances(X_s, X_p)
        #Combine and average the cosine distances of all entries from the relevant section of the primary and secondary trial
        combined_scores = []
        combined_scores.extend(secondary_scores[0])
        combined_scores.extend(primary_scores[0])
        score = numpy.average(combined_scores)
        #If the cosine distance is gless than 0.9 the prediction is entailment   
        if score > 0.9:
            Prediction = "Contradiction"
        else:
            Prediction = "Entailment"
        Results[str(uuid_list[i])] = {"Prediction":Prediction}
    else:
        #If the cosine distance is greater than 0.9 the prediction is contradiction
        score = numpy.average(primary_scores)
        if score > 0.9:
            Prediction = "Contradiction"
        else:
            Prediction = "Entailment"
        Results[str(uuid_list[i])] = {"Prediction":Prediction}


print(Results)
with open("results.json",'w') as jsonFile:
    jsonFile.write(json.dumps(Results,indent=4))


def main():
    gold = dev
    results = Results  
    uuid_list = list(results.keys())

    results_pred = []
    gold_labels = []
    for i in range(len(uuid_list)):
        if results[uuid_list[i]]["Prediction"] == "Entailment":
            results_pred.append(1)
        else:
            results_pred.append(0)
        if gold[uuid_list[i]]["Label"] == "Entailment":
            gold_labels.append(1)
        else:
            gold_labels.append(0)

    f_score = f1_score(gold_labels,results_pred)
    p_score = precision_score(gold_labels,results_pred)
    r_score = recall_score(gold_labels,results_pred)

    print('F1:{:f}'.format(f_score))
    print('precision_score:{:f}'.format(p_score))
    print('recall_score:{:f}'.format(r_score))

if '__main__' == __name__:
    main()


