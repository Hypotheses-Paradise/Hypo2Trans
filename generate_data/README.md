# Hypotheses Paradise: Generation

[[Data]](https://github.com/Hypotheses-Paradise/HP-V0)
[[Paper]]()
[[Colab example]]()

Hypotheses Paradise (HP) is an open-sourced dataset that collects N-best hypotheses and ground-truth transcription from popular ASR datasets.
Here we provide a recipe to generate HP dataset using [Whisper](https://github.com/openai/whisper), a robust large-scale ASR model developed by OpenAI.

## Setup

We used Python 3.8.16 and PyTorch 1.12.1 in our experiments, but the codebase is expected to be compatible with Python 3.7-3.10 and recent PyTorch versions.
You may follow the steps below to install the conda environment:

```bash
# enter your conda env
conda activate your_conda_env

# install required dependencies
pip install -r requirements.txt

# install whisper from source
cd whisper
pip install -e .
```

## Usage

Necessary files are provided in the `whisper` directory. Please kindly follow the steps below to generate HP dataset:

1. open the file `generate_hp_dataset.py` and specify some data paths:

    - `asr_wav`: contains a list of audio files, same as the kaldi-format file "wav.scp";
    - `asr_txt`: contains a list of transcriptions, same as the kaldi-format file "text";
    - `hp_json`: generated HP dataset in .json format;

2. Execute it to generate HP dataset:

    ```bash
    python generate_hp_dataset.py
    ```

## More analysis

We also provide a [Colab example]() to further visualize and analyze our generated HP dataset.

