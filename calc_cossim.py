import argparse
import pathlib

import polars as pl
import torch
from sklearn.metrics.pairwise import cosine_similarity

from resemblyzer import VoiceEncoder, preprocess_wav


def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp_dir", required=True, default=None, type=pathlib.Path)
    parser.add_argument("--out_path", required=True, type=pathlib.Path)
    return parser.parse_args()


def main():
    args = get_arg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = VoiceEncoder(device=device)

    data_path_lst = args.inp_dir.glob("**/*.wav")
    results = []

    for data_path in data_path_lst:
        filename = data_path.parents[0].name
        speaker = data_path.parents[1].name
        date = data_path.parents[2].name
        kind = data_path.stem

        data_path_gt = str(data_path).replace(data_path.stem, "gt")

        wav = preprocess_wav(str(data_path))
        wav_gt = preprocess_wav(str(data_path_gt))

        emb = encoder.embed_utterance(wav)
        emb_gt = encoder.embed_utterance(wav_gt)
        cossim = cosine_similarity(emb.reshape(1, -1), emb_gt.reshape(1, -1))

        results.append([cossim, date, speaker, filename, kind])

    df = pl.DataFrame(
        data=results, schema=["score", "date", "speaker", "filename", "kind"]
    )
    df.write_csv(str(args.out_path))


if __name__ == "__main__":
    main()
