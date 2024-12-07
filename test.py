from pathlib import Path

import numpy as np
import pandas as pd
import torch

from resemblyzer import VoiceEncoder, preprocess_wav


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = VoiceEncoder(device=device)
    encoder.eval()

    df_path = Path("~/dataset/lip/data_split.csv").expanduser()
    audio_dir = Path("~/dataset/lip/wav").expanduser()
    save_dir = Path("~/dataset/lip/emb_fix").expanduser()

    df = pd.read_csv(str(df_path))
    df = df.loc[(df["corpus"].isin(["ATR"])) & (df["data_split"] == "train")]
    speaker_list = list(df["speaker"].unique())
    data_quantity = 100

    for speaker in speaker_list:
        data = df.loc[df["speaker"] == speaker].sample(n=data_quantity, random_state=42)
        emb_list = []

        for row in data.iterrows():
            audio_path = audio_dir / speaker / f'{row[1]["filename"]}.wav'
            wav = preprocess_wav(str(audio_path))
            with torch.no_grad():
                emb = encoder.embed_utterance(wav)
            emb_list.append(emb)

        emb = np.mean(np.array(emb_list), axis=0)
        print(f"{emb.shape}")


if __name__ == "__main__":
    main()
