# The Mystery of Compositional Generalization in Graph-based Generative Commonsense Reasoning
Code and Data for the EMNLP Findings 2024 paper (to appear)

![image](https://github.com/xiyan524/CGGC/blob/main/intro.png)
An instance of Compositional Generalization in Graph-based Commonsense Reasoning (CGGC). A model is expected to solve a test sample (b, composition) that presents an input graph with an unseen combination of relation types (here: *HasA&AtLocation*). The ICL demonstrations of the task in (a), by contrast, show each relation primitive in combination with other relation types, here: *HasA&UsedFor* and *AtLocation&UsedFor*.
 
## Data
#### Data Format
```
{
"concept_set_idx": # sample id in the CommonGen     
"concepts":  # concepts in the concept set    
"refs": [# multiple references
    {"target": # a reference
    "graph": ["nodes", "edges"] # the extracted graph
    "source": # data source (caption or human-annotated)
    "pruned": # pruned edges in the graph
    "pruned_graph": # the extracted graph after pruning
    "graph_label": # graph label used for the compositional generalization
    }, ...]  
}
```

#### Data Download (OneDrive)
**Link**: https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/fuxiyan_mail_nankai_edu_cn/Ek4i_BM-anlMmu0W-gbW6DkBGO42IlLvIndaKSFmHN9M3g?e=eL5OKl

**Files**: verification_train/val [verification]; compos_train/test [compositional generalization]

## Code
For evaluation, please follow these steps:
1. assign the python environment in run-main.sh
2. download the dataset above and assgin the data path in run-main.sh
3. run the script
```
sbatch run-main.sh --max_batch_size batch_size --icl_num --model_name model_name --demonstration_type demo_type
```


## Citations
to appear

