from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from resemblyzer import VoiceEncoder, preprocess_wav

debug = False


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = VoiceEncoder(device=device)

    df_path = Path("~/dataset/jvs_ver1_cut_silence/data_split_fix.csv").expanduser()
    audio_dir = Path("~/dataset/jvs_ver1_cut_silence").expanduser()
    save_dir = Path("~/dataset/jvs_ver1_cut_silence/emb_fix").expanduser()

    df = pd.read_csv(str(df_path))
    speaker_list = list(df["speaker"].unique())
    data_quantity = 100

    for speaker in speaker_list:
        data = df.loc[df["speaker"] == speaker].sample(n=data_quantity, random_state=42)
        emb_list = []

        for row in data.iterrows():
            audio_path = (
                audio_dir
                / speaker
                / row[1]["data"]
                / "wav24kHz16bit"
                / f'{row[1]["filename"]}.wav'
            )
            wav = preprocess_wav(str(audio_path))
            emb = encoder.embed_utterance(wav)
            emb_list.append(emb)

        emb = np.mean(np.array(emb_list), axis=0)

        save_path = save_dir / speaker / "emb.npy"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(save_path), emb)


if __name__ == "__main__":
    main()
