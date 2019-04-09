This project was part of the final project for Stanford's CS 224n Deep Learning for Natural Language Processing

My project was to reimplement the QANet Architecture proposed by Google Brain and CMU in (https://arxiv.org/abs/1804.09541)

To run the code in this repo follow the steps below:

First setup the local environment using
1. conda env create -f environment.yml 
2. source activate squad

Then run python setup.py to get the data in the right folders

Finally run train.py --name <Name of the Run>


Motivation:

Reading comprehension and question answering are critical natural language tasks that many modern NLP models benchmark against. The applications of a good reading comprehension system can extend from digital assistants and chat bots to even medical applications where patients want to know what conditions their symptoms could map to. Previous reading comprehension models have made use of bidirectional attention flows between the context and query as well as encoding representation of the context and query. While these systems perform well they are slow and do not take advantage of recent work in Transformers and models that exclusively convolutions or self attention. Given that most of the new state of the art natural language models are attention based, it seems logical to extend this to reading comprehension. QANet attempts to bring these innovations in neural architecture for a question answering model. QANet proposes to make use of exclusively convolutions and self attentions in the encoders to produce context and query representations. The motivation here is that convolutions capture local structure while the attention captures global structure in the sentence. On top of these encoders, there is a context to query attention layer which implements the bidirectional attention. Finally output probabilities are computed to determine the start and end indices of an answer in the context. 

The key advantage of using convolutions and attention is to improve speed in inference and training. With the increased focus on privacy and running machine learning on the edge, faster models would be preferable.

