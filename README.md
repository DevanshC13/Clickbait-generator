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
