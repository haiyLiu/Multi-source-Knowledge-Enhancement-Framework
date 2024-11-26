# Multi-source-Knowledge-Enhancement-Framework

## Section 1: Prepare the configuration file

- For the FB15k-237 dataset, all parameters are in the config_fb.json.
- For the WN18RR dataset, all parameters are in the config_wn.json.



## Section 2: knowledge from KG and extend knowledge from wikidata

Select the dataset you want to run and change the configuration file of the corresponding data set in config.json.

Then, run the following code:

```
python train.py
```





## Section 3: Query_llm

- For re-ranking instruction

```
python llm/final_llm.py
```

- For select instruction

```
python llm/final_llm_select.py
```

- Merge answers

```
python llm/process_rank_select.py
```

Hint: remember to specify your own OpenAI API key in arguments --api_key.



## Section 4: Evaluation

```
python llm/evaluate.py
```

