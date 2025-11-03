# MLRL Datasets

This directory contains dataset processing instructions for the MLRL experiments.

## arXiv Abstract

This dataset is based on the arXiv scientific papers and is used for the text expansion task based on the abstracts.

### Raw Data
 
Long and structured documents of arXiv scientific papers ([Download](https://drive.google.com/file/d/1b3rmCSIoh6VhD4HKWjI4HOW-cSwcwbeC/view?usp=sharing)).

The files are in JSONlines format, where each line is a JSON object corresponding to one scientific paper. 
The abstract, sections, and body are all sentence tokenized. The JSON objects are in the following format:

```
{ 
  'article_id': str,
  'abstract_text': List[str],
  'article_text': List[str],
  'section_names': List[str],
  'sections': List[List[str]]
}
```

### Processed Dataset

I processed the raw data for the text expansion task with [extract_arXiv_abstract.py](./create_arxiv_abstract.py). The processed dataset is in the same JSONlines format as above, but only contains the `article_id` and `abstract_text` fields, and the abstract length should be **100-300 tokens**. The JSON objects are in the following format:

```
{ 
  'article_id': str,
  'abstract_text': List[str],
  'token_count': int,
}
```

The processed dataset is now available on Huggingface [LovelyBuggies/arXiv_abstract](https://huggingface.co/datasets/LovelyBuggies/arXiv_abstract).


## CoopHumanEval

This dataset is based on the HumanEval benchmark and is used for the code generation task. 