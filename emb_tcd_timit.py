from pathlib import Path
from resemblyzer import preprocess_wav, VoiceEncoder
import random
import torch
import numpy as np
from tqdm import tqdm


def main():
    data_dir = Path('~/tcd_timit').expanduser()
    speaker_list = ['spk1', 'spk2', 'spk3']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = VoiceEncoder(device=device)

    for speaker in speaker_list:
        print(speaker)
        data_dir_spk = data_dir / speaker
        data_path_list = list(data_dir_spk.glob('**/*.wav'))

        emb_list = []
        for data_path in tqdm(data_path_list):
            wav = preprocess_wav(data_path)
            emb = encoder.embed_utterance(wav)
            emb_list.append(emb)
            
        emb = np.mean(np.array(emb_list), axis=0)
        save_dir_spk = data_dir / speaker
        save_dir_spk.mkdir(parents=True, exist_ok=True)
        np.save(str(save_dir_spk / "emb"), emb)


if __name__ == '__main__':
    main()