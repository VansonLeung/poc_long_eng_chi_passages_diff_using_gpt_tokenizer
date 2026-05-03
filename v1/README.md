# GPT Tokenizer Diff POC

This POC compares:

- a long English-Chinese mixed passage
- its data-masked version

It uses Python with `tiktoken` (GPT tokenizer) to:

- tokenize both texts with `cl100k_base`
- diff the token sequences with `difflib.SequenceMatcher`
- emit replacement mappings as JSON

## Files

- `poc_token_diff.py`: tokenizer diff logic and demo runner
- `sample_original.txt`: mixed-language source passage
- `sample_masked.txt`: masked version of the passage
- `replacement_mapping.json`: generated output after running the demo

## Run

```bash
cd v1
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python poc_token_diff.py
```

The script prints the JSON result and writes `replacement_mapping.json`.