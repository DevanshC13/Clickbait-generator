# Clickbait-generator
Bigram language model based clickbait generator (IITK PClub Task 7)
The Clickbait Generator is a web application that uses a trained language model to generate clickbait headlines. The language model is trained on a dataset of clickbait headlines and can generate new headlines based on this training.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.6 or higher
- Flask
- PyTorch

### Installation

1. Clone the repository: `git clone https://github.com/DevanshC13/Clickbait-generator.git`
2. Navigate to the project directory and Open Terminal
3. Run the setup script: `./setup`

### Running the Application

After you've completed the setup, you can run the application with the following command:
`./run`

This will start the Flask server and the application will be accessible at `http://localhost:5000`.

## Usage

The web application has a form where you can specify the maximum length of the headlines, the number of headlines to generate, and the seed for the random number generator.

After you submit the form, the application will generate the specified number of clickbait headlines and display them on the page.

## API

The application also provides an API that you can use to generate clickbait headlines programmatically. The API has one endpoint:

- `POST /api/generate`
  - Parameters:
    - `max_len`: The maximum length of the headlines.
    - `num_headlines`: The number of headlines to generate.
    - `seed`: The seed for the random number generator.
  - Response: A JSON object with a `headlines` property that contains an array of generated headlines.

# Bigram Language model

## Training the Language Model

The language model used in this project is a bigram language model, which was trained based on the methods described in [this video](https://youtu.be/PaCmpygFfXo) by Andrej Karpathy.

A bigram language model is a type of probabilistic language model that predicts the next word in a sentence based on the previous word. It's called a "bigram" model because it considers "bigrams" - pairs of consecutive words in the text.

In the training process, the model goes through the text and calculates the probabilities of different words following each word. These probabilities are then used to generate new text that is similar in style to the training text.

The training process involves several steps:

1. **Data preparation:** The text is split into individual words, and each word is assigned a unique integer ID.

2. **Model initialization:** The model's weights are initialized with random values.

3. **Forward pass:** The model calculates the probabilities of different words following each word in the text.

4. **Loss calculation:** The model calculates the difference between its predictions and the actual next words in the text.

5. **Backward pass:** The model adjusts its weights based on the calculated loss to improve its predictions.

6. **Iteration:** Steps 3-5 are repeated multiple times to further improve the model's predictions.

After the training process, the model can generate new text by choosing the next word based on the calculated probabilities.

For more details about the training process and the bigram language model, you can watch [the video](https://youtu.be/PaCmpygFfXo) by Andrej Karpathy.

### Timeline:

1. Watched the video by Andrej Karpathy, made the following notes:
`xs = stoi[ . e m m a ]
ys = stoi[ e m m a . ]

import torch.nn.functional as F

xenc = F.one_hot(xs, num_classes=27).float()

//construct first neuron

weight of neuron w = torch.randn((27,1))

xenc @ w //vector product = logits

w = torch.randn((27,27))

counts = logits.exp()
probs = counts / counts.sum(1, keepdims=True)
  
// Very random, differentiate the score wrt weights, find
	out gradients, minimize loss

loss -> negative log likelihood

loss = -probs[torch.aragne(5),ys].log().mean()  


// so finally:

//	Forward pass:

xenc = F.one_hot(xs, num_classes=27).float()
logits = xenc @ W
counts = logits.exp()
probs = counts / counts.sum(1, keepdims=True)
loss = -probs[torch.aragne(5),ys].log().mean()

//	Backward pass:

W.grad = None
W.backward()

//	Update:

W.data += -<learning rate> * W.grad
`
2. Trained the model on Google colab for 2000 entries (due to limited resources and time) over 600 epochs varying the learning rate.
3. Documented everything, ClickbaitLogfile.txt contains the logfile for training.



