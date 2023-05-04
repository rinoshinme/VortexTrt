import os
import time
import torch
from ast_models import ASTModel
import torchaudio
import torch.nn.functional as F


class Demo(object):
    def __init__(self, trained_weights, audio_length=512, model_size='base384', method='mean'):
        n_class = 2
        fstride = 10
        tstride = 10
        audio_model = ASTModel(
            label_dim=n_class, fstride=fstride, tstride=tstride, input_fdim=128, input_tdim=audio_length, 
            imagenet_pretrain=False, audioset_pretrain=False, model_size=model_size)
        state_dict = torch.load(trained_weights, map_location='cpu')
        # remove `module.`
        new_state_dict = dict()
        for k, v in state_dict.items():
            newk = k.replace('module.', '')
            new_state_dict[newk] = v
        audio_model.load_state_dict(new_state_dict)

        audio_model.eval()
        self.model = audio_model

        self.target_length = audio_length
        self.melbins = 128
        self.norm_mean = -6.6268077
        self.norm_std = 5.358466
        self.method = method
    
    def aggregate_result(self, outputs):
        if self.method == 'mean':
            outputs = torch.mean(outputs, dim=0)
        else:
            outputs = torch.max(outputs, dim=0)[0]
        return outputs

    def infer(self, audio_path):
        inputs = self.preprocess1(audio_path)
        # print(inputs.shape)
        with torch.no_grad():
            outputs = self.model(inputs)
        outputs = self.aggregate_result(outputs)
        # softmax
        outputs = F.softmax(outputs)
        outputs = outputs.detach().cpu().numpy()
        return outputs
    
    def infer_raw(self, audio_path):
        """
        infer without aggregation.
        """
        inputs = self.preprocess1(audio_path)
        # print(inputs)

        with torch.no_grad():
            outputs = self.model(inputs)
        outputs = outputs.detach().cpu().numpy()
        return outputs
    
    def preprocess1(self, audio_path):
        # process for multiple segments
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform - waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=sr, 
            use_energy=False, window_type='hanning', 
            num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        n_frames = fbank.shape[0]
        if n_frames <= self.target_length:
            p = self.target_length - n_frames
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
            fbank = fbank.unsqueeze(0)
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        else:
            nsegments = n_frames // self.target_length + 1
            offset = (n_frames - self.target_length) // (nsegments - 1)
            clips = []
            for i in range(nsegments):
                start = i * offset
                end = start + self.target_length
                clip = fbank[start:end, :]
                clip = (clip - self.norm_mean) / (self.norm_std * 2)
                clips.append(clip)
            fbank = torch.stack(clips)
        return fbank
    
    def preprocess(self, audio_path):
        fbank = self._wav2fbank(audio_path)
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        fbank = fbank.unsqueeze(0)
        return fbank
    
    def _wav2fbank(self, filename):
        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=sr, 
            use_energy=False, window_type='hanning', 
            num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        n_frames = fbank.shape[0]
        p = self.target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:self.target_length, :]
        return fbank
    
    def to_onnx(self, onnx_path):
        dummy_inputs = torch.randn(1, 3, 224, 224)
        input_names = ['audio_feature']
        output_names = ['outputs']
        torch.onnx.export(self.model, dummy_inputs, onnx_path, verbose=True, 
            input_names=input_names, output_names=output_names)
        
    def test_time(self, audio_path, repeat=5):
        start = time.time()
        for _ in range(repeat):
            out = self.infer_raw(audio_path)
        end = time.time()
        return end - start


if __name__ == '__main__':
    # augmenet
    trained_weights = './augment_audio_model.35.pth'
    model_size = 'base384'
    audio_length = 256

    # original
    # trained_weights = '../weights/audio_model.85.pth'
    # model_size = 'base384'
    # audio_length = 512
    
    demo = Demo(trained_weights, audio_length=audio_length, model_size=model_size)

    audio_path = '../../sample/audio_moan.wav'

    outputs = demo.infer_raw(audio_path)
    print(outputs)
