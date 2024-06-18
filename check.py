from pathlib import Path
import pandas as pd


def main():
    # df_path = Path("~/dataset/lip/data_split.csv").expanduser()
    # audio_dir = Path("~/dataset/lip/wav").expanduser()
    # save_dir = Path("~/dataset/lip/emb_fix").expanduser()

    # df = pd.read_csv(str(df_path))
    # df = df.loc[(df["corpus"].isin(["ATR"])) & (df["data_split"] == "train")]

    # df_path = Path(
    #     "~/dataset/hi-fi-captain_cut_silence/ja-JP/data_split.csv"
    # ).expanduser()
    # audio_dir = Path("~/dataset/hi-fi-captain_cut_silence/ja-JP").expanduser()
    # save_dir = Path("~/dataset/hi-fi-captain_cut_silence/ja-JP/emb_fix").expanduser()

    # df = pd.read_csv(str(df_path))
    # df = df.loc[df["data_split"] == "train"]

    df_path = Path("~/dataset/jvs_ver1_cut_silence/data_split_fix.csv").expanduser()
    audio_dir = Path("~/dataset/jvs_ver1_cut_silence").expanduser()
    save_dir = Path("~/dataset/jvs_ver1_cut_silence/emb_fix").expanduser()

    df = pd.read_csv(str(df_path))
    breakpoint()


if __name__ == "__main__":
    main()
