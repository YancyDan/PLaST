import os
import pandas as pd
from tqdm import tqdm

import librosa
import soundfile as sf

import argparse

SPLITS = ['dev', 'train', 'test']
src_Lan_li = ['en']
tgt_Lan_li = ['de']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-lang", "-s", type=str, required=True,
                        help="source language code")
    parser.add_argument("--tgt-lang", "-t", type=str, required=True,
                        help="target language code")
    return parser.parse_args()

def main():
    args = get_args()
    src_Lan, tgt_Lan = args.src_lang, args.tgt_lang

    os.makedirs(f'./audio/{src_Lan}/clips_16k', exist_ok=True)

    for split in SPLITS:
        excepts = []
        df = pd.read_csv(f'./tsv_orig/covost_v2.{src_Lan}_{tgt_Lan}.{split}.tsv', sep='\t', quoting=3)

        files = df['path'].unique()
        for file in tqdm(files, desc=f"Processing {split}"):
            mp3_file = f'./audio/{src_Lan}/clips/{file}'
            wav_file = f'./audio/{src_Lan}/clips_16k/{file[:-4]}.wav'
            
            try:
                if os.path.exists(wav_file):
                    y, sr = librosa.load(wav_file)
                else:
                    y, sr = librosa.load(mp3_file, sr=16000)
                    sf.write(wav_file, y, sr)
            except:
                # print(f"file error: {wav_file}")
                excepts.append(file)

        df = df[~df['path'].isin(excepts)]
        df['path'] = df['path'].str[:-4] + ".wav"
        df.to_csv(f'./tsv/covost_v2.{src_Lan}_{tgt_Lan}.{split}.tsv', sep='\t', index=False)
        print(f'{split} excepts: {excepts}')

if __name__ == "__main__":
    main()