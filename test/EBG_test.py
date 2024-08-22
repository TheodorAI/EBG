import sys
import time
import warnings
from pathlib import Path
from typing import Optional
import re
import csv
from tqdm import tqdm
import lightning as L
import torch
from torch.nn.utils.rnn import pad_sequence
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from generate import generate
from lit_llama import Tokenizer, LLaMA
from lit_llama.lora import lora
from lit_llama.utils import lazy_load, llama_model_lookup
from scripts.prepare_alpaca import generate_prompt

lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
label_num = {'faithful': 0, 'proud': 0, 'trusting': 0, 'grateful': 0, 'caring': 0, 'hopeful': 0, 'confident': 0, 'excited': 0, 'anticipating': 0, 'surprised': 0, 'sentimental': 0, 'impressed': 0, 'content': 0, 'angry': 0, 'disgusted': 0, 'jealous': 0, 'ashamed': 0, 'anxious': 0, 'lonely': 0, 'sad': 0, 'apprehensive': 0, 'guilty': 0, 'afraid': 0, 'embarrassed': 0}
label_score = {'faithful': 0, 'proud': 0, 'trusting': 0, 'grateful': 0, 'caring': 0, 'hopeful': 0, 'confident': 0, 'excited': 0, 'anticipating': 0, 'surprised': 0, 'sentimental': 0, 'impressed': 0, 'content': 0, 'angry': 0, 'disgusted': 0, 'jealous': 0, 'ashamed': 0, 'anxious': 0, 'lonely': 0, 'sad': 0, 'apprehensive': 0, 'guilty': 0, 'afraid': 0, 'embarrassed': 0}
attri_num = {'positive':0,'negative':0}
attri_score = {'positive':0,'negative':0}
emo_dict = {'faithful': 'positive', 'proud': 'positive', 'trusting': 'positive', 'grateful': 'positive', 'caring': 'positive', 'hopeful': 'positive', 'confident': 'positive', 'excited': 'positive', 'anticipating': 'positive', 'surprised': 'positive', 'sentimental': 'positive', 'impressed': 'positive', 'content': 'positive', 'angry': 'negative', 'disgusted': 'negative', 'jealous': 'negative', 'ashamed': 'negative', 'anxious': 'negative', 'lonely': 'negative', 'sad': 'negative', 'apprehensive': 'negative', 'guilty': 'negative', 'afraid': 'negative', 'embarrassed': 'negative'}
error = 0

def read_csv(filename):
    first_column = []
    second_column = []
    combined_data = set()  # 用于跟踪已经存在的数据
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过首行
        for row in reader:
            combined_row = row[0] + row[1]
            if combined_row not in combined_data:
                first_column.append(row[0])
                second_column.append(row[1])
                combined_data.add(combined_row)
    return first_column, second_column

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

def write_to_file(error_code, dict1, dict2, dict3, dict4, dict5, dict6, filename='./result/result_lora_10_maxpadding.txt'):
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


def main():
    global error,label_score,attri_score
    input: str = ""
    lora_path: Path = Path("out/lora/peft_1_10k/lit-llama-lora-finetuned.pth")
    pretrained_path: Path = Path("checkpoints/lit-llama/7B/lit-llama.pth")
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model")
    quantize: Optional[str] = None
    max_new_tokens: int = 300
    top_k: int = 200
    temperature: float = 0.8

    assert lora_path.is_file()
    assert pretrained_path.is_file()
    assert tokenizer_path.is_file()

    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=1, precision=precision)
    
    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(lora_path) as lora_checkpoint:
        name = llama_model_lookup(pretrained_checkpoint)
        with fabric.init_module(empty_init=True), lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
            model = LLaMA.from_name(name)
            # 1. Load the pretrained weights
            model.load_state_dict(pretrained_checkpoint, strict=False)
            # 2. Load the fine-tuned lora weights
            model.load_state_dict(lora_checkpoint, strict=False)
    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)
    model.eval()
    model = fabric.setup(model)
    tokenizer = Tokenizer(tokenizer_path)

    label_list, question_list = read_csv('./data/evaluation.csv')
    print(len(label_list))
    print(len(question_list))
    # 更新总个数
    for label in label_list:
        label_num[label] = label_list.count(label)

    for key, value in label_num.items():
        emotion = emo_dict.get(key)
        if emotion is not None:
            attri_num[emotion] += value

    print(label_num)
    print(attri_num)

    # 创建空列表用于存储编码后的张量,存储padding
    question_tensors = []
    for i in tqdm(range(len(label_list)),desc="Processing", unit="item"):    
        prompt = question_list[i]
        sample = {"instruction": prompt, "input": input}
        prompt = generate_prompt(sample)
        encoded = tokenizer.encode(prompt, bos=True, eos=True, device=model.device)
        question_tensors.append(encoded)

    max_length = max(len(t) for t in question_tensors)
    question_tensors = pad_sequence(question_tensors, batch_first=True, padding_value=0)
    question_tensors = question_tensors[:, :max_length].contiguous()
    print(max_length)
    # avg_length = int(sum(len(t) for t in question_tensors) / len(question_tensors))
    # question_tensors = pad_sequence(question_tensors, batch_first=True, padding_value=0)
    # question_tensors = question_tensors[:, :avg_length]
    # print(avg_length)
    
    
    for i in tqdm(range(len(label_list)),desc="Processing", unit="item"):    
        # prompt = question_list[i]
        # prompt = question_list[i]
        # sample = {"instruction": prompt, "input": input}
        # prompt = generate_prompt(sample)
        # print(prompt)
        # tokenizer = Tokenizer(tokenizer_path)
        # encoded = tokenizer.encode(prompt, bos=True, eos=True, device=model.device)
        # print(encoded.shape)
        ground_truth = label_list[i]
        attribute_truth = emo_dict[ground_truth]
        t0 = time.perf_counter()
        # print(question_tensors[i].shape)
        print(question_tensors[i])

        output = generate(
            model,
            idx=question_tensors[i],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=tokenizer.eos_id
        )

        t = time.perf_counter() - t0
        output = tokenizer.decode(output)
        response = output.split("### Response:")[1].strip()
        print(response)
        attribute,label = get_label_from_response(response)
        print(attribute,label)
        print(attribute_truth,ground_truth)
        if attribute != None and label != None:
            if attribute == attribute_truth:
                attri_score[attribute_truth]+=1        
                if ground_truth in label:
                    label_score[ground_truth]+=1
        else:
            print('error')
            # print(response)
            error +=1
            label_num[ground_truth] -=1
            attri_num[attribute_truth] -=1
        print(f"\n\nTime for inference: {t:.02f} sec total, {max_new_tokens / t:.02f} tokens/sec", file=sys.stderr)
    attri_rate = calculate_ratio(attri_score, attri_num)
    label_rate = calculate_ratio(label_score, label_num)
    # print(attri_rate)
    write_to_file(error,attri_score,attri_num,attri_rate,label_score,label_num,label_rate)


if __name__ == "__main__":
    from jsonargparse import CLI
    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI(main)
