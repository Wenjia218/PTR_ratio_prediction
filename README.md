# PTR_ratio_prediction

install all notebook packages:

```python
pip install -r requirements.txt
```

The goal of this project is to evaluate the usage of language models to predict PTR ratios across tissues.

Milestones could be:

●	Understand and collect the different sequence features used by our model.

●	Run the model with 10-fold cross validation using the original features. Verify that you get the same results as in the paper.

●	Generate protein embeddings using ESM-2 [5].

●	Replace protein sequence features by the protein embeddings and train a new model.

●	Run RiNALMo [7] on 5’ and 3’ sequences. 

●	Replace 5’ and 3’ sequence features and train a new model.

●	Systematically compare and comment on what you find. Are there stark, statistically significant differences?

●	(Optional) Perform in silico saturation mutagenesis to interpret your results.

