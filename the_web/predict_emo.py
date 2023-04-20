# for the predict
import os
import numpy as np
import torch
from torch import nn
from torch.nn import init
import librosa
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
sample_rate = 48000

augmented_waveforms_temp = []
'''predict the emotion about unload audio'''
if torch.cuda.is_available() == True:
    device = 'cuda'
else:
    device = 'cpu'

def feature_melspectrogram(
        waveform,
        sample_rate,
        fft=1024,
        winlen=512,
        window='hamming',
        hop=256,
        mels=128,
):
    # Produce the mel spectrogram for all STFT frames and get the mean of each column of the resulting matrix to create a feature array
    # Using 8khz as upper frequency bound should be enough for most speech classification tasks
    melspectrogram = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=fft,
        win_length=winlen,
        window=window,
        hop_length=hop,
        n_mels=mels,
        fmax=sample_rate / 2)

    # convert from power (amplitude**2) to decibels
    # necessary for network to learn - doesn't converge with raw power spectrograms
    melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)

    return melspectrogram


def feature_mfcc(
        waveform,
        sample_rate,
        n_mfcc=40,
        fft=1024,
        winlen=512,
        window='hamming',
        # hop=256, # increases # of time steps; was not helpful
        mels=128
):
    # Compute the MFCCs for all STFT frames
    # 40 mel filterbanks (n_mfcc) = 40 coefficients
    mfc_coefficients = librosa.feature.mfcc(
        y=waveform,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=fft,
        win_length=winlen,
        window=window,
        # hop_length=hop,
        n_mels=mels,
        fmax=sample_rate / 2
    )

    return mfc_coefficients


def get_features(waveforms, features, samplerate):
    # initialize counter to track progress
    # process each waveform individually to get its MFCCs
    for waveform in waveforms:
        mfccs = feature_mfcc(waveform, sample_rate)
        features.append(mfccs)
        # print progress
    # return all features from list of waveforms
    return features


def get_waveforms(file):
    # load an individual sample audio file
    # read the full 3 seconds of the file, cut off the first 0.5s of silence; native sample rate = 48k
    # don't need to store the sample rate that librosa.load returns
    waveform, _ = librosa.load(file, duration=5, offset=0.5, sr=sample_rate)
    # waveform, _ = librosa.load(file, duration=3, offset=0.5, sr=sample_rate)

    # make sure waveform vectors are homogenous by defining explicitly
    waveform_homo = np.zeros((int(sample_rate * 5, )))
    waveform_homo[:len(waveform)] = waveform

    # return a single file's waveform
    return waveform_homo

# def augment_waveforms(waveforms, features, emotions, multiples):
def augment_waveforms(waveforms, features, multiples):
    # # keep track of how many waveforms we've processed so we can add correct emotion label in the same order
    # emotion_count = 0
    # # keep track of how many augmented samples we've added
    # added_count = 0
    # # convert emotion array to list for more efficient appending
    # emotions = emotions.tolist()

    for waveform in waveforms:

        # Generate 2 augmented multiples of the dataset, i.e. 1440 native + 1440*2 noisy = 4320 samples total
        augmented_waveforms = awgn_augmentation(waveform, multiples=multiples)

        # compute spectrogram for each of 2 augmented waveforms
        for augmented_waveform in augmented_waveforms:
            # Compute MFCCs over augmented waveforms
            augmented_mfcc = feature_mfcc(augmented_waveform, sample_rate=sample_rate)

            # append the augmented spectrogram to the rest of the native data
            features.append(augmented_mfcc)
            # emotions.append(emotions[emotion_count])
            #
            # # keep track of new augmented samples
            # added_count += 1

        #     # check progress
        #     print(
        #         '\r' + f'Processed {emotion_count + 1}/{len(waveforms)} waveforms for {added_count}/{len(waveforms) * multiples} new augmented samples',
        #         end='')
        #
        # # keep track of the emotion labels to append in order
        # emotion_count += 1

        # store augmented waveforms to check their shape
        augmented_waveforms_temp.append(augmented_waveforms)

    return features

def awgn_augmentation(waveform, multiples=2, bits=16, snr_min=15, snr_max=30):
    # get length of waveform (should be 3*48k = 144k)
    wave_len = len(waveform)

    # Generate normally distributed (Gaussian) noises
    # one for each waveform and multiple (i.e. wave_len*multiples noises)
    noise = np.random.normal(size=(multiples, wave_len))

    # Normalize waveform and noise
    norm_constant = 2.0 ** (bits - 1)
    norm_wave = waveform / norm_constant
    norm_noise = noise / norm_constant

    # Compute power of waveform and power of noise
    signal_power = np.sum(norm_wave ** 2) / wave_len
    noise_power = np.sum(norm_noise ** 2, axis=1) / wave_len

    # Choose random SNR in decibels in range [15,30]
    snr = np.random.randint(snr_min, snr_max)

    # Apply whitening transformation: make the Gaussian noise into Gaussian white noise
    # Compute the covariance matrix used to whiten each noise
    # actual SNR = signal/noise (power)
    # actual noise power = 10**(-snr/10)
    covariance = np.sqrt((signal_power / noise_power) * 10 ** (- snr / 10))
    # Get covariance matrix with dim: (144000, 2) so we can transform 2 noises: dim (2, 144000)
    covariance = np.ones((wave_len, multiples)) * covariance

    # Since covariance and noise are arrays, * is the haddamard product
    # Take Haddamard product of covariance and noise to generate white noise
    multiple_augmented_waveforms = waveform + covariance.T * noise

    return multiple_augmented_waveforms

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        return out


def load_checkpoint(optimizer, model, filename):
    checkpoint_dict = torch.load(filename)
    epoch = checkpoint_dict['epoch']
    model.load_state_dict(checkpoint_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    return epoch


class parallel_all_you_want(nn.Module):
    # Define all layers present in the network
    def __init__(self, num_emotions):
        super().__init__()

        ################ TRANSFORMER BLOCK #############################
        # maxpool the input feature map/tensor to the transformer
        # a rectangular kernel worked better here for the rectangular input spectrogram feature map/tensor
        self.transformer_maxpool = nn.MaxPool2d(kernel_size=[1, 4], stride=[1, 4])

        # define single transformer encoder layer
        # self-attention + feedforward network from "Attention is All You Need" paper
        # 4 multi-head self-attention layers each with 64-->512--->64 feedforward network
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=40,  # input feature (frequency) dim after maxpooling 128*563 -> 64*140 (freq*time)
            nhead=4,  # 4 self-attention layers in each multi-head self-attention layer in each encoder block
            dim_feedforward=512,  # 2 linear layers in each encoder block's feedforward network: dim 64-->512--->64
            dropout=0.4,
            activation='relu'  # ReLU: avoid saturation/tame gradient/reduce compute time
        )

        # I'm using 4 instead of the 6 identical stacked encoder layrs used in Attention is All You Need paper
        # Complete transformer block contains 4 full transformer encoder layers (each w/ multihead self-attention+feedforward)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=4)

        ############### 1ST PARALLEL 2D CONVOLUTION BLOCK ############
        # 3 sequential conv2D layers: (1,40,282) --> (16, 20, 141) -> (32, 5, 35) -> (64, 1, 8)
        self.conv2Dblock1 = nn.Sequential(

            # 1st 2D convolution layer
            nn.Conv2d(
                in_channels=1,  # input volume depth == input channel dim == 1
                out_channels=16,  # expand output feature map volume's depth to 16
                kernel_size=3,  # typical 3*3 stride 1 kernel
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(16),  # batch normalize the output feature map before activation
            nn.ReLU(),  # feature map --> activation map
            nn.MaxPool2d(kernel_size=2, stride=2),  # typical maxpool kernel size
            nn.Dropout(p=0.3),  # randomly zero 30% of 1st layer's output feature map in training

            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=16,
                out_channels=32,  # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),  # increase maxpool kernel for subsequent filters
            nn.Dropout(p=0.3),

            # 3rd 2D convolution layer identical to last except output dim
            nn.Conv2d(
                in_channels=32,
                out_channels=64,  # expand output feature map volume's depth to 64
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )
        ############### 2ND PARALLEL 2D CONVOLUTION BLOCK ############
        # 3 sequential conv2D layers: (1,40,282) --> (16, 20, 141) -> (32, 5, 35) -> (64, 1, 8)
        self.conv2Dblock2 = nn.Sequential(

            # 1st 2D convolution layer
            nn.Conv2d(
                in_channels=1,  # input volume depth == input channel dim == 1
                out_channels=16,  # expand output feature map volume's depth to 16
                kernel_size=3,  # typical 3*3 stride 1 kernel
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(16),  # batch normalize the output feature map before activation
            nn.ReLU(),  # feature map --> activation map
            nn.MaxPool2d(kernel_size=2, stride=2),  # typical maxpool kernel size
            nn.Dropout(p=0.3),  # randomly zero 30% of 1st layer's output feature map in training

            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=16,
                out_channels=32,  # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),  # increase maxpool kernel for subsequent filters
            nn.Dropout(p=0.3),

            # 3rd 2D convolution layer identical to last except output dim
            nn.Conv2d(
                in_channels=32,
                out_channels=64,  # expand output feature map volume's depth to 64
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )

        ################# FINAL LINEAR BLOCK ####################
        # Linear softmax layer to take final concatenated embedding tensor
        #    from parallel 2D convolutional and transformer blocks, output 8 logits
        # Each full convolution block outputs (64*1*8) embedding flattened to dim 512 1D array
        # Full transformer block outputs 40*70 feature map, which we time-avg to dim 40 1D array
        # 512*2+40 == 1064 input features --> 8 output emotions
        # self.fc1_linear = nn.Linear(512 * 2 + 40, num_emotions)
        self.fc1_linear = nn.Linear(1832, num_emotions)

        ### Softmax layer for the 8 output logits from final FC linear layer
        self.softmax_out = nn.Softmax(dim=1)  # dim==1 is the freq embedding

    # define one complete parallel fwd pass of input feature tensor thru 2*conv+1*transformer blocks
    def forward(self, x):
        ############ 1st parallel Conv2D block: 4 Convolutional layers ############################
        # create final feature embedding from 1st convolutional layer
        # input features pased through 4 sequential 2D convolutional layers
        conv2d_embedding1 = self.conv2Dblock1(x)  # x == N/batch * channel * freq * time

        # flatten final 64*1*4 feature map from convolutional layers to length 256 1D array
        # skip the 1st (N/batch) dimension when flattening
        # conv2d_embedding1 = torch.flatten(conv2d_embedding1, start_dim=1)

        ############ 2nd parallel Conv2D block: 4 Convolutional layers #############################
        # create final feature embedding from 2nd convolutional layer
        # input features pased through 4 sequential 2D convolutional layers
        conv2d_embedding2 = self.conv2Dblock2(x)  # x == N/batch * channel * freq * time

        # flatten final 64*1*4 feature map from convolutional layers to length 256 1D array
        # skip the 1st (N/batch) dimension when flattening
        # conv2d_embedding2 = torch.flatten(conv2d_embedding2, start_dim=1)

        y = torch.cat([conv2d_embedding1, conv2d_embedding2])  # torch.Size([64, 64, 1, 14])
        sa = ScaledDotProductAttention(d_model=14, d_k=64, d_v=1, h=64)
        sa = sa.to(device)
        y = sa(y, y, y)
        y = torch.flatten(y, start_dim=1)

        ########## 4-encoder-layer Transformer block w/ 64-->512-->64 feedfwd network ##############
        # maxpool input feature map: 1*40*282 w/ 1*4 kernel --> 1*40*70
        x_maxpool = self.transformer_maxpool(x)

        # remove channel dim: 1*40*70 --> 40*70
        x_maxpool_reduced = torch.squeeze(x_maxpool, 1)

        # convert maxpooled feature map format: batch * freq * time ---> time * batch * freq format
        # because transformer encoder layer requires tensor in format: time * batch * embedding (freq)
        x = x_maxpool_reduced.permute(2, 0, 1)

        # finally, pass reduced input feature map x into transformer encoder layers
        transformer_output = self.transformer_encoder(x)

        # create final feature emedding from transformer layer by taking mean in the time dimension (now the 0th dim)
        # transformer outputs 64*140 (freq embedding*time) feature map, take mean of all columns i.e. take time average
        transformer_embedding = torch.mean(transformer_output, dim=0)  # dim 40x70 --> 40
        a = transformer_embedding.shape[0]
        # #([n, 40])
        ############# concatenate freq embeddings from convolutional and transformer blocks ######
        # concatenate embedding tensors output by parallel 2*conv and 1*transformer blocks
        y1 = y[:a, ]
        y2 = y[a:, ]
        complete_embedding = torch.cat([y1, y2, transformer_embedding], dim=1)

        ############# concatenate freq embeddings from convolutional and transformer blocks ######
        # concatenate embedding tensors output by parallel 2*conv and 1*transformer blocks
        # complete_embedding = torch.cat([conv2d_embedding1, conv2d_embedding2, transformer_embedding], dim=1)

        ######### final FC linear layer, need logits for loss #########################
        output_logits = self.fc1_linear(complete_embedding)

        ######### Final Softmax layer: use logits from FC linear, get softmax for prediction ######
        output_softmax = self.softmax_out(output_logits)

        # need output logits to compute cross entropy loss, need softmax probabilities to predict class
        return output_logits, output_softmax


from pydub import AudioSegment
import wave
import io


def mp3_to_wav(mp3_path, wav_path):
    """
    mp3 转 wav
    :param mp3_path: 输入的 mp3 路径
    :param wav_path: 输出的 wav 路径
    :return:
    """
    # 读取 mp3
    fp = open(mp3_path, 'rb')
    data = fp.read()
    fp.close()
    # 读取
    aud = io.BytesIO(data)
    sound = AudioSegment.from_file(aud, format='mp3')
    raw_data = sound._data

    # 写入 wav
    nframes = len(raw_data)
    f = wave.open(wav_path, 'wb')
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(48000)
    f.setnframes(nframes)
    f.writeframes(raw_data)
    f.close()


def predict(unload_audio_name):

    waveforms = [get_waveforms(unload_audio_name)]
    features_test=[]
    # specify multiples of our dataset to add as augmented data
    multiples = 2

    waveform = np.array(waveforms)
    X_test=waveform
    features_test = get_features(X_test, features_test, sample_rate)

    features_test= augment_waveforms(X_test, features_test, multiples)

    filename = 'features+labels+test.npy'

    # 处理输入的音频
    X_test = np.expand_dims(features_test, 1)

    N, C, H, W = X_test.shape
    X_test = np.reshape(X_test, (N, -1))
    # X_test = scaler.transform(X_test)
    ss = StandardScaler()
    res_data = ss.fit_transform(X_test)
    X_test = ss.inverse_transform(res_data)
    X_test = np.reshape(X_test, (N, C, H, W))

    f = open(filename, 'wb')
    np.save(f, X_test)
    f = open(filename, 'rb')
    X_test = np.load(f)

    X_test_tensor = torch.tensor(X_test, device=device).float()

    # device = 'cuda'

    load_folder = ''

    # pick the epoch to load
    # epoch = '029'
    # model_name = f'parallel_all_you_wantFINAL_anima_4-{epoch}.pkl'

    # make full load path
    # load_path = os.path.join(load_folder, model_name)
    load_path =f"parallel_all_you_wantFINAL_anima4-029.pkl"

    ## instantiate empty model and populate with params from binary
    model = parallel_all_you_want(4).to(device)

    # model = model.cuda()
    # # 扔到cuda去
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-3, momentum=0.8)

    load_checkpoint(optimizer, model, load_path)

    # set model to validation phase i.e. turn off dropout and batchnorm layers
    model.eval()

    # get the model's predictions on the validation set
    output_logits, output_softmax = model(X_test_tensor)
    predictions = torch.argmax(output_softmax, dim=1)

    # emotions_dict = {'0': 'neutral', '1': 'happy', '2': 'sad', '3': 'angry'}
    emotions_dict = {'4': 'neural：这段话是没啥情感的',
                     '1': 'happy：真不错呐',
                     '2': 'sad：听上去有点悲伤，少听点，^v^',
                     '3': 'angry：攻击力有点强'}

    emo=predictions[0]
    emo=str(emo)
    emo=int(emo[7])+1
    emo=str(emo)
    emo_label = emotions_dict[emo]

    return(emo_label)

    #
    # # Convert 4D test feature set array to tensor and move to GPU
    # X_test_tensor = torch.tensor(X_test, device=device).float()
    # # Convert 4D test label set array to tensor and move to GPU
    # y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)

if __name__ == '__main__':
    unload_audio_name=r"D:\test_no_english\test1.wav"
    emo=predict(unload_audio_name)
    print('the predict emo is ')
    print(str(emo))