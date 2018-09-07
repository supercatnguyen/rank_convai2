# Introduction
- Team name: Cats'team 
- Model name: lstmSim
# Installation 
- Run on python 3
- clone from github: "git clone https://github.com/supercatnguyen/rank_convai2.git"
- Install Parlai: "cd rank_convai2", "python setup.py develop"
- Install Pytorch (can run with pytorch 0.4), torchtext
- Uncompress file:  "unzip parlai/agents/k_model/w2vec.out.zip" 
# How to Run
- To run: "cd rank_convai2/projects/convai2/baselines/k_model"; "python eval_hits.py".

# Result
- On valid dataset the result in command line should be: 
```
EPOCH DONE
finished evaluating task convai2:self using datatype valid
{'hits@1': 0.434, 'exs': 7801}

```