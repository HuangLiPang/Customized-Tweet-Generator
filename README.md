# Customized Tweet Generator from Uploaded Images

The repo is modified from [PyTorch Flask API](https://github.com/avinassh/pytorch-flask-api-heroku).

<img src="https://i.imgur.com/MQw82v6.png" width="250">

## Description

This application generates a customized tweet (post) from the image uploaded from the user. The model is built with [Pytorch](https://pytorch.org/) which takes an image and generates a customized post. The backend services are developed using [Flask](https://flask.palletsprojects.com/en/1.1.x/).

In this demo, we used the dataset from Taiwanese president's official [Twitter](https://twitter.com/iingwen) to train the model. The dataset is crawled from [Twitter API](https://developer.twitter.com/en/docs/twitter-api) and it includes the tweets from 2010 to 2020 with total 3200 data. You can find the [dataset](./dataset) and the [code](./model) for training the model.

For more information, please check the [paper](./docs/Customized_Tweet_Generator_from_Uploaded_Images.pdf) and the introduction [video](https://www.youtube.com/watch?v=E0c_ig-5MFs).

## Customized Post vs Original Post
<img src="https://i.imgur.com/qd9LjSz.png" width="250">  <img src="https://i.imgur.com/5mohmdz.png" width="250">

## Local Deployment

Clone this repository:

    git clone https://github.com/HuangLiPang/Customized-Tweet-Generator.git

Install the dependecies from `requirements.txt`:

    pip install -r requirements.txt

Download the model and the vocab from [here](https://www.cs.virginia.edu/~lh5jv/models/) and add them under the model directory in this repository.

    cd model
    wget https://www.cs.virginia.edu/~lh5jv/models/state_dict.pth
    wget https://www.cs.virginia.edu/~lh5jv/models/vocab.pth

Run the server:

    cd ..
    python app.py

If you have questions, feel free to post an issue [here](https://github.com/HuangLiPang/Customized-Tweet-Generator/issues).

## License

The mighty MIT license. Please check `LICENSE` for more details.
