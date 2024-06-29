# EvolvePro
PLM based active learning model for protein engineering
# directed_evolution

The order of analysis is specified below. This is a rough repo, but overall describes key steps undertaken in the analysis. 


### esm-extract:

`extract.sh` is a basic OpenMind compatible bash file for running the `extract.py` file released with esm. It relies on the fasta file to mean embeddings of mutants. The output format is a .pth file for each mutant, where each file is named after the substitution.

To make the data more workable for downstream tasks, `concatenate.sh` calls on `concatenate.py` to generate a single csv of mean esm embeddings. The results of this are saved in `results_means/csvs`



### top-layer-metrics

The top_layer.py contains python code for top-layer RL

