Source for datasets:

* Got all of these datasets from https://github.com/JoshEngels/SAE-Probes/blob/main/data/probing_datasets_MASTER.csv, running own cleaning pipeline

To do: add dataset type to each col (qa, classification, etc.)
Missing from SAE probing paper:
* 39_arith_infix.csv 
* 40_arith_rpn.csv  (how to produce these - I think they generated themselves)
* only using sciq train right now, and also phys_reasoning train rn. Could use more of it in og since will do splits myself
* duplicates in name and save_name
* truthful qa generation vs multiple choice?? not sure of difference yet!
* add notes section to main.csv for each possibly?
* wikidata sex or gender, should I be using name or text ??? this is so curs-ed https://www.dropbox.com/scl/fo/lvajx9100jsy3h9cvis7q/ACU8osTw0FCM_X-d8Wn-3ao/cleaned_data?dl=0&preview=56_wikidatasex_or_gender.csv&rlkey=tq7td61h1fufm01cbdu2oqsb5&subfolder_nav_tracking=1. I mean the wikidata datasets are all so sus....
* note: using the atleete wiki, added football and basketball and baseball players YAY, in total 5 classes of players, added all of those
* phys reasoning train labels are separate death!
* left: from truthful qa to the rest of the og place ones, skip og place ones, then to the end (around 50 done, 100 to go)
* not using proglangtest.csv -- seems only to have like 70 data points or smt,so no
* the textinfo datasets seem like crap? like what is contains digit? like they all have a number somewhere? lmaoo maybe look deeper into these
* for context ans dataset, I could also make this binary classification version and multiclass version both. binary is easier but more logical, multiclass could just be highly random shit or smt
* context_question_type dataset might be interesting if you just have question text, or also context + question text -> probe to question type
* for reasoning-tf I'm also just using train. **Maybe I should add a handler that merges train test and validatoin if there are multiple or smt, for better training. How I can do this, also with labels one, is have a folder for some datasets with multiple files for them this might be super nice eventually!**
* commonsense dataset: just used the train set for now, and there were atleast 2 more, test, and test_hard
* unclear how to use utilitarianism, but leo gao does use it 
* I have more arrow files I haven't used so far! Keep these in mind for sure.
* only using train for openqa
* lots of sets in compound_words.csv are maybe lost because they are tokenized. Should I download Wes Sparse Probing tokenizer and decode them?
* 89,cancer_cat.csv,cancer_cat,Cancer Doc Classification,cancer_cat.csv,a,"0",Multiclass Classification,a,"0",simple, faield 
* fix 92 test one at some point lol

Thoughts? If making this easy for probing library, upload all your datasets to hugging face so you can just use HF datasets~ For now, don't load in datasets not on Hugging Face! Can you upload a folder of datasets or something? Ah even on hugging face you want to be able to parse them, this is taking portions of other peoples' cant do that



One thing: if the files have no headers, we create headers where column names are numbers.


## Handler Schemas

### `simple` Handler
- **Purpose:**  
  - Extracts and (optionally) transforms columns from tabular datasets as specified in `main.csv`.
- **Expected Source Data:**  
  - CSV file with columns matching `Probe from` and `Probe to`.
- **main.csv columns used:**
  - `Probe from`: Comma-separated list of source columns for the prompt.
  - `Probe to`: Comma-separated list of target columns.
  - `probe from extraction`: Python expression (string) for extracting/transforming the prompt.
  - `probe to extraction`: Python expression (string) for extracting/transforming the target.
  - `handler`: Should be set to `simple`.
- **Output columns:**
  - `prompt`: String generated from `probe from extraction`.
  - `prompt_len`: Length of the prompt string.
  - `target`: Value from `probe to extraction`.
- **Example Use Cases:**  
  - Standard tabular datasets (e.g., country names, ages, categories).

### `sciq_balancer` Handler
- **Purpose:**  
  - Formats multiple-choice question datasets so each prompt is paired with either the correct answer or a randomly selected incorrect answer (50/50), and includes a binary correctness label.
- **Expected Source Data:**  
  - CSV/Parquet/Jsonl file with the following columns:
    - A column containing the question text.
    - A column for the correct answer.
    - Some columns for incorrect answer options.
- **main.csv columns used:**
  - `Probe from`: Comma-separated list of question column, and then incorrect answer columns.
  - `Probe to`: Correct answer column.
  - `handler`: Should be set to `sciq_balancer`.
  - *All other extraction fields are ignored for this handler.*
- **Output columns:**
  - `prompt`: Formatted as `Q: {question} A: {answer}` (with `{answer}` correct or incorrect).
  - `prompt_len`: Length of the prompt string.
  - `label`: `1` if the answer is correct, `0` if it is incorrect.
- **Example Use Cases:**  
  - Science MCQ datasets, quiz data, or any Q&A where you want balanced correct/wrong answers.

### `arithmetic_label` Handler

- **Purpose:**  
  - Formats multiple-choice question (MCQ) datasets where:
    - The question text and answer choices are specified in `Probe from`.
    - The target label is specified in `Probe to`.
  - The output `prompt` is the question plus all choices with labels, and `target` is the correct label.

- **Expected Source Data:**  
  - CSV, Parquet, or JSONL file.
  - A column containing questions
  - A way to access choices text and choices label (whether in same or different column)
  - A column containing the correct answer label

- **main.csv columns used:**
  - `Probe from`: Question column, choices text column, choices label column
  - `Probe to`: Correct label column
  - `handler`: Set to `arithmetic_label`
  - *All other extraction fields are ignored.*

- **Output columns:**
  - `prompt`: The question plus formatted answer options, e.g. `Calculate 5 + 91 = A.97 B.95 C.106 D.96`
  - `prompt_len`: Length of the prompt string
  - `target`: The correct answer label (e.g., `D`)

- **Example Use Cases:**  
  - MCQ math, science, or quiz datasets with choices as arrays and labels as the classification target.





