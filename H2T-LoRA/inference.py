import os
import sys
import editdistance
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import re
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
from argparse import ArgumentParser
import json
import logging


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass
load_8bit = True


def calculate_wer(pre, ref):
    wer_score = editdistance.eval(pre, ref) / len(ref)
    return wer_score


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True, help='<your_test_data>.json')
    args = parser.parse_args()

    prompter = Prompter("alpaca")
    tokenizer = LlamaTokenizer.from_pretrained('yahma/llama-13b-hf')
    model = LlamaForCausalLM.from_pretrained(
        'yahma/llama-13b-hf',
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map='auto',
    )
    model = PeftModel.from_pretrained(
        model,
        args.ckpt_path,
        torch_dtype=torch.float16,
        device_map={'': 0},
    )

    model.config.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    if not load_8bit:
        model.half()
    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)


    def alpaca_infer(input1, input2=None):
        if input2 is not None:
            prompt = prompter.generate_prompt(input=input1, input2=input2)
        else:
            prompt = prompter.generate_prompt(input=input1)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=256,
            stream_output=False,
            prompter=prompter,
            model=model,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=256,
            )

        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return prompter.get_response(output)


    with open(args.test_data_path, 'r') as f:
        test_data = json.load(f)

    before = 0
    after = 0

    ignore = 0

    for i in range(len(test_data)):

        best_hypo = test_data[i]['input1']
        input2 = test_data[i]['input2']
        ground_truth = test_data[i]['output']
        prediction = alpaca_infer(input1=best_hypo, input2=input2)

        prediction = re.sub('</s>', '', prediction).lower()
        prediction = re.sub(r'[^\w\s]|[\d]', '', prediction)
        best_hypo = re.sub(r'[^\w\s]|[\d]', '', best_hypo)
        ground_truth = re.sub(r'[^\w\s]|[\d]', '', ground_truth)
        prediction = re.sub(r'\n+.+', '', prediction)

        try:
            wer_prediction = calculate_wer(prediction.split(), ground_truth.split())
            wer_best_hypo = calculate_wer(best_hypo.split(), ground_truth.split())
        except Exception:
            ignore += 1
            continue

        before = before + wer_best_hypo
        after = after + wer_prediction
        if wer_best_hypo != wer_prediction:
            print('before:::', best_hypo)
            print('after :::', prediction)
            print('answer:::', ground_truth)
            print('before score', wer_best_hypo)
            print('after score', wer_prediction)


    print('before', before / (len(test_data) - ignore))
    print('after', after / (len(test_data) - ignore))


