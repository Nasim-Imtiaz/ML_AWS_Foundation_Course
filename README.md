# AWS Machine Learning Foundations Nanodegree Program

#### Amazon Codeguru - Code Development
improve code quality and identify most expensive lines of code. 

#### Amazon Forecast - Forecasting
acheive high accuracy forecast on product demands, resource needs or financial performance. 

#### Amazon Fruad Detector - Fraud
identify potentially frauduient activity. 

#### Amazon Lex - Chatbots
publishes voice or text bots.

#### Amazon Lookout for Metrics - Anomaly Detection
automatically detect and diagonose anomalies. 

#### Amazon Monitron - Industrial AI
predicts machines failures.

#### Amazon Personalize - Personalization
deliver highly customized recommendations. 

#### Amazon Polly - Speech
turns text into lifelike speech allowing you to create applications that talk.

#### Amazon Recognition - Vision
fast and accurate face search of private images. 

#### Amazon Transcribe Medical - Health AI
turns medical speech to text.

#### Amazon Textract - Text
automatically extracts texts, handwriting, data from scanned document. 

#### Artificial Intelligence
AI refers to the broad capability of machines to perform activities using human-like intelligence. 

#### AWS DeepComposer
composite device powered by generative AI that creates a medoly that transforms into a completely original song. 

#### AWS DeepLens
deep learning-enabled video camera. 

#### AWS DeepRacer
an autonomous race car designed to test reinforcement learning models by racing a physical track. 

#### Bag of words
extract features from the text. It counts how many times a word appeares in a document.

#### Categorical Label 
has a discrete set of possible values, such as "is cat" and "is not a cat". 

#### Clustering
Unsupervised learning task helps to determine if there are any naturally occuring groupings in the data. 

#### Coefficient of determination or R-squared
This meansures how well-observed outcomes are actually predicted by the model, based on the porportion of total variation of outcomes. 

#### Component of Machine Learning
* Machine Learning Model - extremely generic program(or block of code), made specific by the data used to train it. It is used to solve different problems. ( *generic problem made specific by data* )
* Model Training Algorithm - Model training algorithms work through an interactive process. It thinks about the changes that need to be made and make those changes and repeat these process. ( *an iterative process fitting a generic model to soecific data* )( *iteratively update model parameters to minimize some loss function* )
* Model Inference algorithm - generates prediction using the trained model. ( *process to use a trained model to solve a task* )

#### Continuous Label
doesn't have a discrete set of possible values, which means possibly an unlimited of possibilities. 

#### Data Portion in Machine Learning workflow
* Data Collection
* Data Inspection
  * Outliers Detection
  * Missing or Incomplete Data
  * Transform Your Dataset
* Summary Statistics
  * Trends in the data
  * Scale of the data
  * Shape of the data
* Data Visualization

#### Data Vectorization
process that converts non-numeric data into a numerical format so that it can be ised by a machine learning model. 

#### Deep Learning
Extremely popular and powerful, a modern approach based around a conceptual model of how the human brain functions. The model is composed of collections of nueronos connected together by weights. The process of training involves finding values for each weight. 
* FFNN - Feed Forward Neural Network structures neurons in a series of layers, with each neuron in a layer containing weights to all neurons in the previous layer. 
* CNN - Convolutional Nueral Network represents nested filters over grid-organized data. Most commonly used type of model when processing images. 
* RNN/LSTM - Recuurent neural Network and the Long Short-term Memory model type are structured efficiently represenet for loops in traditional computing, collecting state while iterating over some object. Used for processing sequences of data. 
* Transformer - A more modern replacement for RNN/LSTMs, the transformer architecture enables training over larger datasets involving sequences of data. 

#### Hyperparameters 
are settings on the model which are not changed during training but can affect how quickly or how reliably the model trains, such as the number of clusters the model should identify. 

#### Inference 
ready to generate predictions on real-world problems using unseen data in the field using the trained model. 

#### Input Layer
the first layer in a neural ntwork. This layer receives all data that passes through the neural network.

#### Label Data
refers to data that already contains the solution. 

#### Linear Models
describe the relationship between a set of inpus numbers and a set of output numbers through a linear function. Linear models are fast to train. Classification tasks often use a strongly related logistic model.  

#### Log Loss 
seeks to calculate how uncertain your model is about the predictions it is generating. 

#### Loss Function
Measurement of how close the model is to its goal. 

#### Machine Learning
Which allows computers to automatically learn and improve from experience without being explicitly programmed to do so. Using machine learning computer learn to ***discover patterns*** and ***make predictions***. It combines statistics, applied math and computer science. 

#### Mean Absolute Error (MAE)
measured by taking the average of the avsolute difference between the actual values and the predictions.

#### Model
Model inspects data to discover patterns. Then human use the pattern learn by the model to gain new understandings or make predictions. 

#### model Accuracy
is the fraction of predictions a model gets right. 

#### Model Parameters 
Configuration that changes how the model behaves 

#### Model Selection
determines which model or models to use. 

#### Reinforcement Learning 
Reinforcement Learning learns through consequences of actions in an environment. Ex. training pet. The algorithm figures out which actions to take in a situation to maximize a reward on the way to reaching a specific goal. 

#### Root Mean Square Error (RMSE)
values with large error receive a higher penalty. RMSE takes the square root of the average difference between the prediction and the actual value.

#### Silhouette coefficient 
score from -1 to 1 describing the clusters found during modeling. A score near zero indicates ovewrlapping clusters, and the scores less than zero indicate data point assigned to incorrect clusters. A score approaching 1 indicates successful identification of discrete non-overlapping clusters. 

#### Steps in Machine Learning
* Defining the problem
* Building the dataset
* Training the model
 * Splitting Dataset
* Evaluating the model
* Using the model

#### Stop Words
list of words removed by NLP tools when building datset. 

#### Supervised Learning
Supervised Learning is a type of machine learning technique in which every training sample from the dataset has a corresponding label or output value associated with it. Supervised learning algorithms learn to predict labels or output values. ***Regression*** and ***Classification*** are supervised leraning. 

#### Test Dataset
The test dataset will be used during model evaluation.

#### Tree-based Models
Lean to categorize or regress by building an extremely large structure of nested if/else blocks, splitting thw world into different regions at ech if/else block. Training determines exactly where these splits happen and what value is assigned at each leaf region. Ex. XGBBoost Model.

#### Unlabeled Data
means you don't need to provide the model with any kind of label or solution while the model is being trained. 

#### Unsupervised Learning 
Unsupervised Learning is a type of machine learning technique in which there is no labels for training data. Unsupervised algorithms try to learn the underlying patterns or distributions that govern the data. 

### Resources
* [Udacity - AWS foundational Course](https://www.udacity.com/course/aws-machine-learning-foundations--ud090)
* Classical Models Library - scikit-learn
* Common metrics - sklearn
* Deep Learning Library - mxnet, tensorflow, pytorch
