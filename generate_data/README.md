# Hypotheses Paradise: Whisper N-Best Hypotheses Generation

[[Data]](https://github.com/Hypotheses-Paradise/HP-V0)
[[Paper]]()
[[Colab example]](https://drive.google.com/file/d/1fE6xfmc0uFNyBQLsuQSYBnQ17ZZfP7pD/view?usp=sharing)

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

Necessary files are provided in the `whisper` directory. Please run the following command to generate HP dataset:

```bash
python generate_hp_dataset.py --asr_wav /path/to/wav.scp --asr_txt /path/to/text --hp_json /path/to/hp.json
```

- `asr_wav`: contains a list of audio paths, where each line indicates a sample, e.g. "utt_id_1 /path/to/1.wav";
- `asr_txt`: contains a list of transcriptions, where each line indicates a sample, e.g. "utt_id_1 i have a dream";
- `hp_json`: generated HP dataset in .json format, where each item indicates a sample, e.g. {'input': ['i have a dream', 'i have dream', 'i had a dream', 'i have a drink', 'i have a drill'], 'output': 'i have a dream'}, here the 'input' contains N-best hypotheses and the 'output' is ground-truth transcription;

## Visualizations and analysis

We present a [Colab example](https://drive.google.com/file/d/1fE6xfmc0uFNyBQLsuQSYBnQ17ZZfP7pD/view?usp=sharing) to further visualize and analyze our generated HP dataset.

