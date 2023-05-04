import librosa
import torch
import torchaudio
import numpy as np
import onnxruntime as ort


class OnnxInfer(object):
    def __init__(self, model_path):
        self.sr = 16000
        self.melbins = 128
        self.target_length = 256
        self.norm_mean = -6.6268077
        self.norm_std = 5.358466

        # load onnx model
        self.sess = ort.InferenceSession(model_path, None)
    
    def preprocess(self, audio):
        waveform, sr = librosa.load(audio_path, sr=self.sr)
        if len(waveform.shape) == 2:
            waveform = np.transpose(waveform, (1, 0))
        else:
            waveform = np.expand_dims(waveform, axis=0)

        waveform = torch.from_numpy(waveform)

        # get filter banks
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=self.sr, 
            use_energy=False, window_type='hanning', 
            num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        print(fbank.shape)

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

        return fbank.cpu().numpy()

    def infer(self, audio_path):
        inputs = self.preprocess(audio_path)
        print(inputs)
        outputs = self.sess.run(['outputs'], {'audio_feature': inputs})
        return outputs


if __name__ == '__main__':
    onnx_path = './ast-base384-len128.onnx'
    audio_path = '../../sample/audio_moan.wav'

    demo = OnnxInfer(onnx_path)
    demo.infer(audio_path)
