# LLMs for Building Codes

This repository contains the code for an AI-based system capable of answering natural language questions about the National Building Code of Canada.

## Organization

* The `search.py` file implements the experiment to measure the effectiveness of the search algorithms.
* The `generate.py` file implements the experiment to measure the effectiveness of the LLMs. The `likelihood.py` and `similarity.py` files implement the actual calculation of the associated metrics.
* The `data_generator.py` file implements the process used to generate the testing dataset.
* The `embeddings.py` file implements the process used to train a doc2vec-based embeddings model.

Unfortunately, the testing dataset cannot be published due to copyright constraints.

## Citation

```
@article{doi:10.1061/JCCEE5.CPENG-6037,
    author = {Isaac Joffe and George Felobes and Youssef Elgouhari and Mohammad Talebi Kalaleh and Qipei Mei and Ying Hei Chui },
    title = {The Framework and Implementation of Using Large Language Models to Answer Questions about Building Codes and Standards},
    journal = {Journal of Computing in Civil Engineering},
    volume = {39},
    number = {4},
    year = {2025},
    doi = {10.1061/JCCEE5.CPENG-6037},
    url = {https://ascelibrary.org/doi/abs/10.1061/JCCEE5.CPENG-6037},
}
```
