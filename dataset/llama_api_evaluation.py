from openai import OpenAI
import re
import csv
from tqdm import tqdm
import os
import replicate
os.environ["REPLICATE_API_TOKEN"] = ""

label_num = {'faithful': 0, 'proud': 0, 'trusting': 0, 'grateful': 0, 'caring': 0, 'hopeful': 0, 'confident': 0, 'excited': 0, 'anticipating': 0, 'surprised': 0, 'sentimental': 0, 'impressed': 0, 'content': 0, 'angry': 0, 'disgusted': 0, 'jealous': 0, 'ashamed': 0, 'anxious': 0, 'lonely': 0, 'sad': 0, 'apprehensive': 0, 'guilty': 0, 'afraid': 0, 'embarrassed': 0}
label_score = {'faithful': 0, 'proud': 0, 'trusting': 0, 'grateful': 0, 'caring': 0, 'hopeful': 0, 'confident': 0, 'excited': 0, 'anticipating': 0, 'surprised': 0, 'sentimental': 0, 'impressed': 0, 'content': 0, 'angry': 0, 'disgusted': 0, 'jealous': 0, 'ashamed': 0, 'anxious': 0, 'lonely': 0, 'sad': 0, 'apprehensive': 0, 'guilty': 0, 'afraid': 0, 'embarrassed': 0}
attri_num = {'positive':0,'negative':0}
attri_score = {'positive':0,'negative':0}
emo_dict = {'faithful': 'positive', 'proud': 'positive', 'trusting': 'positive', 'grateful': 'positive', 'caring': 'positive', 'hopeful': 'positive', 'confident': 'positive', 'excited': 'positive', 'anticipating': 'positive', 'surprised': 'positive', 'sentimental': 'positive', 'impressed': 'positive', 'content': 'positive', 'angry': 'negative', 'disgusted': 'negative', 'jealous': 'negative', 'ashamed': 'negative', 'anxious': 'negative', 'lonely': 'negative', 'sad': 'negative', 'apprehensive': 'negative', 'guilty': 'negative', 'afraid': 'negative', 'embarrassed': 'negative'}
error = 0
attri_wrong = {}
label_wrong = {}
inpu = ''

def read_csv(filename):
    first_column = []
    second_column = []
    combined_data = set() 
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  
        for row in reader:
            combined_row = row[0] + row[1]
            if combined_row not in combined_data:
                first_column.append(row[0])
                second_column.append(row[1])
                combined_data.add(combined_row)
    return first_column, second_column

def create_COT(question):
    prompt1 = '''<Your task is to analyse a sentence, giving a explanation of the process of your thought,select the emotion expressed in this sentence from the table below.
    positive_list = ['faithful', 'proud', 'trusting', 'grateful', 'caring', 'hopeful', 'confident', 'excited', 'anticipating', 'surprised', 'sentimental', 'impressed', 'content']
    negative_list = ['angry', 'disgusted', 'jealous', 'ashamed', 'anxious','lonely', 'sad', 'apprehensive', 'guilty', 'afraid', 'embarrassed']>
    
    For example,giving the sentence {I've been hearing strange noises around the house at night},your response is:
    ##Analysing the attribute of the sentence, it was (negative)
    ##Finding words in the text that convey emotions $noises,strange$
    ##Consider the context of the text, $When someone hears strange things at night$
    ##Choosing two answers from ['angry', 'disgusted', 'jealous', 'ashamed', 'anxious','lonely', 'sad', 'apprehensive', 'guilty', 'afraid', 'embarrassed'],The answers from the list are (afraid|anxious)
    
    For example,giving the sentence {It is computer science. I am very happy of this achievement and my family is very proud},your response is:
    ##Analyzing the attributes of the sentence, it is (positive).
    ##Finding words in the text that convey emotions: $happy,proud$
    ##Considering the context of the text: $Achieving something in an area that brings happiness and pride to both the speaker and their family$
    ##Choosing two answers from ['faithful', 'proud', 'trusting', 'grateful', 'caring', 'hopeful', 'confident', 'excited', 'anticipating', 'surprised', 'sentimental', 'impressed', 'content'],The answers from the list are (excited|proud)
    '''
    prompt2 = "For example,giving the sentence {"+question + "}, your response is:"
    prompt = prompt1+prompt2
    return prompt


def gpt_output(prompt):
    input = {
        "top_p": 1,
        "system_prompt": "Play as a careful thinker to follow the instructions",    
        "prompt": prompt,
        "max_new_tokens": 300
    }

    output = replicate.run(
        "meta/llama-2-70b-chat",
        input=input
    )
    response = "".join(output)
    return response


def get_label_from_response(response):
    pattern = r'\((.*?)\)'  
    matches = re.findall(pattern, response)
    if len(matches) >= 2:
        return matches[0], matches[-1]
    else:
        return None, None

def calculate_ratio(attri_score, attri_num):
    result_dict = {}

    for key in attri_score:
        ratio = attri_score[key] / attri_num[key]
        result_dict[key] = round(ratio, 3)
    return result_dict

def write_to_file(error_code, dict1, dict2, dict3, dict4, dict5, dict6, filename='result_llama70.txt'):
    with open(filename, 'w') as file:
        file.write("Error: " + str(error_code) + "\n\n")
        file.write("attri_score:\n")
        for key, value in dict1.items():
            file.write(f"{key}: {value}\n")
        file.write("\n")
        file.write("attri_num:\n")
        for key, value in dict2.items():
            file.write(f"{key}: {value}\n")
        file.write("\n")
        file.write("attri_rate:\n")
        for key, value in dict3.items():
            file.write(f"{key}: {value}\n")
        file.write("\n")
        file.write("label_score:\n")
        for key, value in dict4.items():
            file.write(f"{key}: {value}\n")
        file.write("\n")
        file.write("label_num:\n")
        for key, value in dict5.items():
            file.write(f"{key}: {value}\n")
        file.write("\n")
        file.write("label_rate:\n")
        for key, value in dict6.items():
            file.write(f"{key}: {value}\n")
    print(f"Data has been written to {filename}")



if __name__ == '__main__':
    label_list, question_list = read_csv('evaluation_test.csv')
    print(len(label_list))
    print(len(question_list))
    
    for label in label_list:
        label_num[label] = label_list.count(label)

    for key, value in label_num.items():
        emotion = emo_dict.get(key)
        if emotion is not None:
            attri_num[emotion] += value

    print(label_num)
    for i in tqdm(range(len(label_list)),desc="Processing", unit="item"):         
        question = question_list[i]
        ground_truth = label_list[i]
        attribute_truth = emo_dict[ground_truth]
        prompt = create_COT(question)
        # print(prompt)
        response = gpt_output(prompt)
        # print(response)
        attribute,label = get_label_from_response(response)
        print(attribute,label)
        print(attribute_truth,ground_truth)
        if attribute !=None and label!= None:
            if attribute == attribute_truth:
                attri_score[attribute_truth]+=1        
                if ground_truth in label:
                    label_score[ground_truth]+=1
        else:
            print('error')
            print(response)
            error +=1
            label_num[ground_truth] -=1
            attri_num[attribute_truth] -=1

    attri_rate = calculate_ratio(attri_score, attri_num)
    label_rate = calculate_ratio(label_score, label_num)
    # print(attri_rate)
    write_to_file(error,attri_score,attri_num,attri_rate,label_score,label_num,label_rate)
    