import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.nn.utils.rnn as rnn


class CoattentionNet(nn.Module):
    ''' Model class for Hierarchical Co-Attention Net
    '''

    def __init__(self, vocabulary_size, num_classes, embedding_dimension=512, k=30):
        ''' Constructor for Hierarchical Co-Attention Net model. Predicts an answer to a question about 
        an image using the Hierarchical Question-Image Co-Attention for Visual Question Answering (Lu et al, 2017) paper.
        Parameters:
            vocabulary_size: int; Number of words in the vocabulary.
            num_classes: int; Number of output classes.
            embedding_dimension: int; Embedding dimension.
            k; int; 
        '''

        super().__init__()

        self.embed = nn.Embedding(vocabulary_size, embedding_dimension)  # embedding for each word in the vocabulary. Each tensor being of size 512

        # question convolutions
        self.unigram_conv = nn.Conv1d(embedding_dimension, embedding_dimension, 1, stride=1, padding=0)
        self.bigram_conv  = nn.Conv1d(embedding_dimension, embedding_dimension, 2, stride=1, padding=1, dilation=2)
        self.trigram_conv = nn.Conv1d(embedding_dimension, embedding_dimension, 3, stride=1, padding=2, dilation=2)

        self.max_pool = nn.MaxPool2d((3, 1))
        self.lstm = nn.LSTM(input_size=embedding_dimension, hidden_size=embedding_dimension, num_layers=3, dropout=0.4)
        self.tanh = nn.Tanh()

        # weights for feature extraction and co-attention
        self.W_b = nn.Parameter(torch.randn(embedding_dimension, embedding_dimension))
        self.W_v = nn.Parameter(torch.randn(k, embedding_dimension))
        self.W_q = nn.Parameter(torch.randn(k, embedding_dimension))
        self.W_hv = nn.Parameter(torch.randn(k, 1))
        self.W_hq = nn.Parameter(torch.randn(k, 1))

        # weights for conjugation
        self.W_w = nn.Linear(embedding_dimension, embedding_dimension)
        self.W_p = nn.Linear(embedding_dimension * 2, embedding_dimension)
        self.W_s = nn.Linear(embedding_dimension * 2, embedding_dimension)

        # weights for classification
        self.fc = nn.Linear(embedding_dimension, num_classes)

    
    def forward(self, image_tensor, question_tensor):
        ''' Forward propagation.
        Parameters:
            image_tensor: pytorch tensor; The image.
            ques_tensor: pytorch tensor; The question.
        Returns:
            output: pytorch tensor; The probability of the final answer.
        '''

        # Image: batch_size x 512 x 196 from the image encoder.
        question, lens = rnn.pad_packed_sequence(question_tensor)  # pads multiple sequences of differing lengths
        question = question.permute(1, 0)  # Question: batch_size x len_of_question
        words = self.embed(question).permute(0, 2, 1)  # Words: batch_size x len_of_question x 512

        unigrams = torch.unsqueeze(self.tanh(self.unigram_conv(words)), 2) # batch_size x 512 x len_of_question
        bigrams  = torch.unsqueeze(self.tanh(self.bigram_conv(words)), 2)  # batch_size x 512 x len_of_question
        trigrams = torch.unsqueeze(self.tanh(self.trigram_conv(words)), 2) # batch_size x 512 x len_of_question
        words = words.permute(0, 2, 1)  # Words: batch_size x len_of_question x 512

        phrase = torch.squeeze(self.max_pool(torch.cat((unigrams, bigrams, trigrams), 2)))
        phrase = phrase.permute(0, 2, 1)  # Phrase: batch_size x len_of_question x 512

        # pass the question through an LSTM
        hidden_input = None  # hidden_input is None for the first time.
        phrase_packed = nn.utils.rnn.pack_padded_sequence(torch.transpose(phrase, 0, 1), lens)  # packs multiple padded sequences with the given lengths.
        sentence_packed, hidden_input = self.lstm(phrase_packed, hidden_input)
        sentence, _ = rnn.pad_packed_sequence(sentence_packed)
        sentence = torch.transpose(sentence, 0, 1)  # Sentence: batch_size x len_of_question x 512

        # Feature extraction
        v_word, q_word = self.parallel_co_attention(image_tensor, words)  # word-based image-text co-attention
        v_phrase, q_phrase = self.parallel_co_attention(image_tensor, phrase)  # phrase-based image-text co-attention
        v_sentence, q_sentence = self.parallel_co_attention(image_tensor, sentence)  # sentecne-based image-text co-attention

        # Classification
        h_w = self.tanh(self.W_w(q_word + v_word))
        h_p = self.tanh(self.W_p(torch.cat(((q_phrase + v_phrase), h_w), dim=1)))
        h_s = self.tanh(self.W_s(torch.cat(((q_sentence + v_sentence), h_p), dim=1)))

        output = self.fc(h_s)

        return output
    

    def parallel_co_attention(self, V, Q):
        ''' Parallel Co-Attention of Image and text.
        Parameters:
            V: pytorch tensor; Extracted image features.
            Q: pytorch tensor; Extracted question features.
        Returns:
            v: pytorch tensor; Attention vector for the image features.
            q: pytorch tensor; Attention vector for question features.
        '''

        # V: batch_size x 512 x 196, Q: batch_size x length_of_question x 512

        C = self.tanh(torch.matmul(Q, torch.matmul(self.W_b, V))) # batch_size x length_of_question x 196

        H_v = self.tanh(torch.matmul(self.W_v, V) + torch.matmul(torch.matmul(self.W_q, Q.permute(0, 2, 1)), C))                            # batch_size x k x 196
        H_q = self.tanh(torch.matmul(self.W_q, Q.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_v, V), C.permute(0, 2, 1)))           # batch_size x k x length_of_question

        a_v = fn.softmax(torch.matmul(torch.t(self.W_hv), H_v), dim=2) # batch_size x 1 x 196
        a_q = fn.softmax(torch.matmul(torch.t(self.W_hq), H_q), dim=2) # batch_size x 1 x length_of_question

        v = torch.squeeze(torch.matmul(a_v, V.permute(0, 2, 1))) # batch_size x 512
        q = torch.squeeze(torch.matmul(a_q, Q))                  # batch_size x 512

        return v, q


if __name__ == "__main__":
    model = CoattentionNet(10000, 1000)
    print(model)