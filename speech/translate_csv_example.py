#!/usr/bin/env python3

from modules.tokens import TokenUpdater
from modules.translate import Translator
import os
import pandas as pd
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('token_updater').setLevel(logging.DEBUG)
    logging.getLogger('translator').setLevel(logging.DEBUG)
    oauth_token = os.environ['OAUTH_TOKEN']
    folder_id = os.environ['FOLDER_ID']
    token_updater = TokenUpdater(oauth_token = oauth_token) 
    translator = Translator(folder_id=folder_id, 
                            token_getter=token_updater,
                            target_lang='ru')
    df = pd.read_csv('tasks.csv', sep=',')
    df['GPT response'] = translator.translate_batch(df['GPT response'].tolist())['text']
    df['Base prompt'] = translator.translate_batch(df['Base prompt'].tolist())['text']
    df['Task'] = translator.translate_batch(df['Task'].tolist())['text']
    df.to_csv('tasks_translated.csv', index=False)