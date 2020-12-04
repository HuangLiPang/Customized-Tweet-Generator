# Customized Tweet Generator from Uploaded Images

The repo is modified from [PyTorch Flask API](https://github.com/avinassh/pytorch-flask-api-heroku).

## Description
This application focused on generating customized tweet (post) from the image the user uploaded. The model is built with [Pytorch](https://pytorch.org/) which takes an image and generate a customized post. The backend services are developed using [Flask](https://flask.palletsprojects.com/en/1.1.x/).

In this demo, we used the dataset from Taiwanese president's official [Twitter](https://twitter.com/iingwen) to train the model. The dataset is crawled from [Twitter API](https://developer.twitter.com/en/docs/twitter-api). You can find the dataset [here](./dataset) and the code for training the model [here](./model).

## Requirements

Install them from `requirements.txt`:

    pip install -r requirements.txt

## Model and Vocab

Please download the model and the vocab from [here](https://www.cs.virginia.edu/~lh5jv/models/) and add them under the model directory in this repository.

## Local Deployment

Run the server:

    python app.py

## License

The mighty MIT license. Please check `LICENSE` for more details.
