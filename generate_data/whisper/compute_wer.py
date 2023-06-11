import whisper
import jiwer
import os
import numpy as np
import torch
import pandas as pd
import whisper
import torchaudio
from tqdm.notebook import tqdm
from whisper.normalizers import EnglishTextNormalizer

normalizer = EnglishTextNormalizer()

hyp = []
f_hyp=open('hyp.txt', 'r')
for line in f_hyp.readlines():
    text = line.strip()
    if len(text) == 0:
        text = '<unk>'
    hyp.append(text)
f_hyp.close()

ref = []
f_ref=open('ref.txt', 'r')
for line in f_ref.readlines():
    text = normalizer(line.strip())
    if len(text) == 0:
        text = '<unk>'
    ref.append(text)
f_ref.close()

wer = jiwer.wer(ref, hyp)
print(f"WER: {wer * 100:.2f} %")

