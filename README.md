# incas-phase2b-concern

## How to build project

* Then download model files from [here](https://drive.google.com/file/d/1qsJMOoCvzWka86n6aWl2yc7ILhbfBiKW/view?usp=sharing). Please keep the file 'annotate.py' and the folder `incas_tuned_model_ih` in the same directory.

The directory should look like this:

````
incas-israel-hamas-concern
├── incas_tuned_model_2b --place models here
├── Readme.md
├── annotate.py
├── requirements.txt
├── small_sample.jsonl
└── prompt_inference.txt

````
## How to run
```
# Install torch
CPU (win): pip3 install torch torchvision torchaudio
CPU (Linux): pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
(option) GPU: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other lib
pip install -r requirements.txt

# Run concern model
python annotate.py --file your_file_name

# (option) Run concern model on GPU
python annotate-gpu.py --file your_file_name

PS. If using the gpu version, please the GPU VRAM > 30GB, such as A100, A40, A6000.
```

The input file parameter is a single dictionary file.

The return value is a dictionary whose keys are message IDs and values are lists of DARPA-defined annotation instances.

## Example
```
python annotate.py --file small_sample.jsonl
```

input:
```
{{"id": "f35e3a661579ae30f2f5bd69b081ab9a9295a709", "contentText": "RT @amyklobuchar People need to know this: Sen. Tommy Tuberville has now held up 301 military postings. He has even put a hold on the head of the Indo-Pacific command. \u201cThis is a gift to China, and it\u2019s a gift that keeps giving day in and day out\u2026\u201d https://t.co/rlaz1dSh7i"}
```

output:
```
{"concern": {'Concern/US Military': 1, 'Concern/Defense': 0, 'Concern/Territorial Sovereignty': 0, 'Concern/Domestic Economy': 0, 'Concern/Trade': 0, 'Concern/Insurgency Separatism': 0, 'Concern/Crime': 0, 'Concern/NoneOther': 0}}
```

