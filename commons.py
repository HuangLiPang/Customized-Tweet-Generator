import io
import os.path
from PIL import Image
import random

import pytorch_lightning as pl
from torch.nn import functional as F
from torch import nn
import torch
from torchvision import transforms
import torchvision.models as models
import torchtext


class TextDecoder(nn.Module):
    def __init__(self, input_size: int, state_size: int, vocab_size: int):
        super(TextDecoder, self).__init__()
        self.state_size = state_size
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.rnnCell = nn.LSTMCell(input_size, state_size, bias=True)
        self.predictionLayer = nn.Linear(state_size, vocab_size)
        self.init_weights()

    def dummy_input_state(self, batch_size):

        return (torch.zeros(batch_size, self.state_size), torch.zeros(batch_size, self.state_size))

    def init_weights(self):

        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.predictionLayer.bias.data.fill_(0)
        self.predictionLayer.weight.data.uniform_(-0.1, 0.1)

    def forward(self, input_state, current_token_id):
        # Embed the input token id into a vector.
        embedded_token = self.embedding(current_token_id)

        # Pass the embedding through the RNN cell.
        h, c = self.rnnCell(embedded_token, input_state)

        # Output prediction.
        prediction = self.predictionLayer(F.relu(h))

        return prediction, (h, c)


class ImageEncoder(nn.Module):
    # Encode images using Resnet-152
    def __init__(self, encoding_size: int):
        super(ImageEncoder, self).__init__()
        self.base_network = models.resnet152(pretrained=True)
        self.base_network.fc = nn.Linear(
            self.base_network.fc.in_features, encoding_size)
        self.bn = nn.BatchNorm1d(encoding_size, momentum=0.01)
        self.init_weights()

    def init_weights(self):

        self.base_network.fc.weight.data.normal_(0.0, 0.02)
        self.base_network.fc.bias.data.fill_(0)

    def forward(self, image):

        with torch.no_grad():

            x = self.base_network.conv1(image)
            x = self.base_network.bn1(x)
            x = self.base_network.relu(x)
            x = self.base_network.maxpool(x)

            x = self.base_network.layer1(x)
            x = self.base_network.layer2(x)
            x = self.base_network.layer3(x)
            x = self.base_network.layer4(x)

            x = self.base_network.avgpool(x)
            x = torch.flatten(x, 1)

        featureMap = self.base_network.fc(x)
        featureMap = self.bn(featureMap)
        return featureMap


class ImageCaptioner(pl.LightningModule):
    def __init__(self, textTokenizer, val_data=None, embedding_size=512, state_size=1024):
        super(ImageCaptioner, self).__init__()
        self.vocabulary_size = len(textTokenizer.vocab)
        self.padding_token_id = textTokenizer.vocab.stoi["<pad>"]

        self.val_data = val_data

        # Create image encoder and text decoder.
        self.image_encoder = ImageEncoder(state_size)
        self.text_decoder = TextDecoder(embedding_size,
                                        state_size,
                                        self.vocabulary_size)

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.padding_token_id)

        self.init_image_transforms()
        self.text_tokenizer = textTokenizer

        self.image_encoder_learning_rate = 1e-4
        self.text_decoder_learning_rate = 1e-3

    def init_image_transforms(self):
        # Create image transforms using standard Imagenet-based model transforms.
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.image_train_transform = \
            transforms.Compose([transforms.Resize(256),
                                transforms.RandomCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize])

        self.image_test_transform = \
            transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize])

    # Predict text given image -- input text is for "teacher forcing" only.
    def forward(self, image, text, lengths, teacher_forcing=1.0):
        # Keep output scores for tokens in a list.
        predicted_scores = list()

        # Encode the image.
        encoded_image = self.image_encoder(image)

        # Grab the first token in the sequence.
        # print(text.shape)
        start_token = text[:, 0]  # This should be the <start> symbol.

        # Predict the first token from the start token embedding
        # and feed the image as the initial state.
        # let first input state = None
        token_scores, state = self.text_decoder(
            (encoded_image, encoded_image), start_token)
        predicted_scores.append(token_scores)

        # Iterate as much as the longest sequence in the batch.
        # minus 1 because we already fed the first token above.
        # minus 1 because we don't need to feed the end token <end>.
        for i in range(0, max(lengths) - 2):
            if random.random() < teacher_forcing:
                current_token = text[:, i + 1]
            else:
                _, max_token = token_scores.max(dim=1)
                current_token = max_token.detach()  # No backprop.
            token_scores, state = self.text_decoder(state, current_token)
            predicted_scores.append(token_scores)

        # torch.stack(,1) forces batch_first = True on this output.
        return torch.stack(predicted_scores, 1), lengths

    def training_step(self, batch, batch_idx, optimizer_idx):
        images, texts, lengths = batch

        # Compute the predicted texts.
        predicted_texts, _ = self(images, texts, lengths,
                                  teacher_forcing=1.0)

        # Define the target texts.
        # We have to predict everything except the <start> token.
        target_texts = texts[:, 1:].contiguous()

        # Use cross entropy loss.
        loss = self.criterion(predicted_texts.view(-1, self.vocabulary_size),
                              target_texts.view(-1))
        self.log('train_loss', loss, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        images, texts, lengths = batch

        predicted_texts, _ = self(images, texts, lengths,
                                  teacher_forcing=0.0)

        target_texts = texts[:, 1:].contiguous()

        loss = self.criterion(predicted_texts.view(-1, self.vocabulary_size),
                              target_texts.view(-1))
        self.log('val_loss', loss, on_epoch=True)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        print('Validation loss %.2f' % loss_mean)

        return {'val_loss': loss_mean}

    def training_epoch_end(self, outputs):

        loss_mean = torch.stack([x['loss'] for x in outputs[0]]).mean()
        print('Training loss %.2f' % loss_mean)

    def configure_optimizers(self):
        return [torch.optim.SGD(list(self.image_encoder.base_network.fc.parameters()) +
                                list(self.image_encoder.bn.parameters()),
                                lr=self.image_encoder_learning_rate),
                torch.optim.Adam(self.text_decoder.parameters(),
                                 lr=self.text_decoder_learning_rate)], []


def load_vocab():
    base_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(base_path, "model/vocab_remove_url.pth")
    vocab = torch.load(path)
    return vocab


def generate_caption(model, image, max_length=128):
    image = image.reshape(1, 3, 224, 224)
    encoded_image = model.image_encoder(image)

    token_scores, state = model.text_decoder(
        (encoded_image, encoded_image), torch.tensor(2).reshape(-1))
    words = ['<start>']
    for i in range(max_length):
        token_idx = torch.argmax(token_scores.view(-1))
        token = textTokenizer.vocab.itos[token_idx.item()]
        words.append(token)
        if token_idx.item() == 3:
            break
        token_scores, state = model.text_decoder(
            state, torch.tensor(token_idx).reshape(-1))
    return ' '.join(words[1:-1])


textTokenizer = torchtext.data.Field(sequential=True,
                                     init_token="<start>", eos_token="<end>",
                                     pad_token="<pad>", unk_token="<unk>",
                                     batch_first=True)
textTokenizer.vocab = load_vocab()


def get_model():
    model = ImageCaptioner(textTokenizer)
    base_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(base_path, "model/state_dict_remove_url.pth")
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])
    # transforms.Normalize(
    #     [0.485, 0.456, 0.406],
    #     [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)
