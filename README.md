# $AAPL stock sentiment analysis

This project uses three different approaches in order to try and classify financial text data into two specific categories: ‘Bullish’ and ‘Bearish’. 

## Table of Contents
- [Motivation](#motivation)
- [Usage](#usage)
- [Results](#results)
- [Questions](#questions)


## Motivation <a name=“motivation”></a>
Being able to predict stock price movement is a paramount goal in the financial world. StockTwits is a social media platform, similar to X (Twitter) designed for sharing thoughts on stocks, crypto and markets between other users. Being able to analyse the sentiment in these discussions would provide very valuable insights into the market sentiment trend which in turn may be correlated to stock market movements. 
 
In this project, I have performed sentiment analysis on any post related to the stock $AAPL, which have been extracted from StockTwits from January 2021. The benefit of using StockTwits is that it has user-initiated sentiment data; the users have the choice to upload a label on their post of either ‘Bullish’ or ‘Bearish’. 

## Usage <a name=“usage”></a>
To use the dataset in this notebook please
1. Use the dl_intro environment
2. Download the [Data](AAPL_2021.csv) in this GitHub repository
3. Upload the dataset file to the Jupiter Notebook environment 
4. Run the code to read the dataset into a Pandas DataFrame

## Results <a name=“results”></a>
The results of the project and more details on the choice of approaches are in a separate file for your reference. The file is in [Results](Results.md).

## Questions <a name=“questions”></a>
1.	List five different tasks that belong to the field of natural language processing.
	- Text classification
	- Named entity recognition
	- Question answering
	- Text generation
	- Summarisation
2.	What is the fundamental difference between econometrics/statistics and supervised machine learning
	- Econometrics aims to estimate unobservable parameters and tests hypotheses on them, putting more emphasis on causality. Whereas supervised machine learning predicts observable things by building predictive models using labelled data, prioritising predictive accuracy to new data. 
	- Some amount of bias is ok in ML but in econometrics we want consistency and unbiasedness
3.	Can you use stochastic gradient descent to tune the hyperparameters of a random forest. If not, why?
	- Stochastic gradient descent (SGD) is an optimisation algorithm used to train models in supervised learning and random forests are an example of an ensemble learner built on decision trees. To use SGD the hyperparameters need to be continuous and the loss function differentiable. However, random forests have discrete hyperparameters e.g. number of trees, and thus are non-differentiable.
4.	What is imbalanced data and why can it be a problem in machine learning?
	- Imbalanced data is when the distribution of categories or classes in a dataset is skewed. This leads to the issue in ML that models can be biased towards the majority class and by just predicting the majority outcome, can obtain a very high accuracy and high F1 majority score.  
5.	Why are samples split into training and test data in machine learning?
	- A sample is split into training and test data in order to train the model on a subset, and then evaluate its performance on the other subset. Splitting the data creates a way to assess how well the model performs on the unseen test data. This also helps overcome the problem of the model memorising the training data but failing to generalise the new data; overfitting.
6.	Describe the pros and cons of word and character level tokenization.
	- Word level tokenisation has the benefit that it preserves the meaning of words and are more easily interpretable, but it struggles when words are used that are variations or typos and are unseen before. Character level tokenisation has the benefit of a smaller vocabulary size and is much better at handling rare or unseen words. However, it lacks the semantic information as the tokens may not fully capture the meaning of the word and the tokenised texts are very long. 
7.	Why does fine-tuning usually give you a better performing model than feature extraction?
	- Fine tuning usually performs better than feature extraction because it enables the model to adapt to the new task by retraining all the model parameters on the target dataset. This allows the model to use the knowledge from pre-training and apply that to the new dataset without need for extensive training data and thus mitigating overfitting. Feature extraction trains the parameters of the classification model and freezes the model parameters which may not be applicable to the new task, resulting in suboptimal performance. 
8.	What are advantages over feature extraction over fine-tuning
	- Feature extraction requires less data for the task since the pretrained model has already learnt from the training data. Also, as only the final classification layers are trained on the new data it is computationally more efficient during inference and thus still performs well on a CPU. Whereas fine-tuning is very slow without GPU.
9.	Why are neural networks trained on GPUs or other specialized hardware?
	- Neural networks are trained on GPUs due to the parallel processing capabilities. This implies that the floating point calculations are done simultaneously instead of sequentially. This speeds up the complex mathematical computations that are involved in deep learning models, which makes training faster.
10.	How can you write pytorch code that uses a GPU if it is available but also runs on a laptop that does not have a GPU.
	- ```python
        Import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device
        ```
11.	How many trainable parameters would the neural network in this video have if we remove the second hidden layer but leave it otherwise unchanged.
	- We would be left with an input layer with 764 neurons, a first hidden layer with 16 neurons and an output layer with 10 neurons. By removing the second hidden layer we would be reducing 16*10=160 weights and 10 biases associated with these connections. Therefore, there are 12730 parameters.
12.	Why are nonlinearities used in neural networks? Name at least three different nonlinearities.
	- Nonlinearities are used in neural networks to better approximate more intricate patterns in the data by introducing nonlinearity in the network’s activation function. Otherwise our model could be simplified to one linear model. Three examples are ReLU, Sigmoid and Tanh
13.	Some would say that softmax is a bad name. What would be a better name and why?
	- A softmax converts any real-valued vector into a vector of valid probabilities. A better name would be softargmax as an argmax takes a vector as input and returns a one-hot vector of the maximum value, whereas the softargmax is a softened version which is differentiable everywhere.  
14.	What is the purpose of DataLoaders in pytorch?
	- DataLoaders in pytorch allow for efficient and convenient data handling for ML tasks. It makes it easy to loop over batches in the data and abstracts away the mechanics of shuffling and batching so that it is easier to train models. 
15.	Name a few different optimizers that are used to train deep neural networks
	- Stochastic gradient descent (SGD), SGD + momentumAdam (Adaptive Moment Estimation), RMSProp 
16.	What happens when the batch size during the optimization is set too small?
	- A small batch size during optimsation is more memory efficient, but noise is increased which implies that there is more randomness in gradient estimates and destabilises the training process. It also increases the risk of overfitting and also may slow down convergence due to an increase in the number of iterations as the updates are noisy.
17.	What happens when the batch size diring the optimization is set too large?
	- When the batch size is set too large during optimisation, the main challenge is the computational power that would be needed. The large amount of memory needed may require very powerful GPUs. Furthermore, the model may not be generalisable as it may not capture the very fine patterns in the data.
18.	Why can the feed-forward neural network we implemented for image classification not be used for language modelling?
	- The FNN used for image classification processes fixed size input and cannot be used for language modelling as it does not process sequentially which is what is needed for language modelling. Furthermore, images have fixed dimensions and thus expect fixed input sizes, whereas text sequences have variable lengths. 
19.	Why is an encoder-decoder architecture used for machine translation (instead of the simpler encoder only architecture we used for language modelling)
	- Machine translation requires understanding and converting text between languages, demanding both source language comprehension and target language generation. Encoder-decoder is used for these tasks as it can handle inputs and outputs that both consist of variable length sequences (bidirectional processing). Therefore, it is suited to generating new sentences depending on a given input. 
20.	Is it a good idea to base your final project on a paper or blogpost from 2015? Why or why not?
	- It is not a good idea as the transformer architecture was introduced in 2017 in the paper ‘Attention is all you need’. Before this, NLP tasks were using RNNs and other models which struggled with handling complex language understanding.
21.	Do you agree with the following sentence: To get the best model performance, you should train a model from scratch in Pytorch so you can influence every step of the process.
	- I do not fully agree with this. Training a model from scratch in pytorch would give more control and customisation, but it would not yield the best performance. A pre-trained model has been trained on vast datasets and therefore fine-tuning allows for the model to retain this knowledge but adapts to the specific task you need. This is much more data and resource efficient and with a high quality pre-trained model, will result in better performance.
22.	What is an example of an encoder-only model?
	- BERT is an encoder-only model 
23.	What is the vanishing gradient problem and how does it affect training?
	- The vanishing gradient problem arises when gradients become very small during backpropagation which impedes effective training. The small gradients result in slower convergence and poor generalisation. This usually happens with the sigmoid activation function and the problem can be mitigated using ReLU insated.
24.	Which model has a longer memory: RNN or Transformer?
	- The transformer model has a longer memory than RNN. RNNs struggle to capture long-term dependencies due to exploding or vanishing gradients. Whereas transformers use various mechanisms to store information on the context and dependencies between elements in a sequence. 
25.	What is the fundamental component of the transformer architecture?
	- The fundamental component of the transformer architecture is the self-attention mechanism. This allows for each element in a sequence to focus on other elements’ importance in the same sequence. This process enables the model to capture contextual dependencies making it much more effective at NLP tasks than previous architectures.
