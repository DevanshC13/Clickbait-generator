from flask import Flask, render_template, request, flash
from main import prepare_data, SampleData
import torch

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])





def index():
    if request.method == 'POST':
        try:
            max_len = int(request.form['max_len'])
            num_headlines = int(request.form['num_headlines'])
            seed = int(request.form['seed'])
            if max_len < 1 or num_headlines < 1 or seed < 0 or max_len == None or num_headlines == None :
                raise ValueError
            # Here you would use your model to generate the clickbait headlines
            # For now, let's just echo back the input
            W, train_data, test_data = initialize()
            headlines = SampleData(test_data, W, seed=seed, num_samples=num_headlines, max_len=max_len)
            print(headlines)
        except ValueError:
            flash('Invalid input. Please enter positive integers for max length and number of headlines.')
            return render_template('index.html')
        return render_template('index.html', headlines=headlines)
    else:
        return render_template('index.html')

def initialize():
    # ML initialisation
    token_ids, vocab, vocab_size = prepare_data('train1.csv')
    W = torch.load('model_weights4434.pth')
    train_data, test_data = token_ids, token_ids
    return W, train_data, test_data

if __name__ == '__main__':
    app.run(debug=True)
