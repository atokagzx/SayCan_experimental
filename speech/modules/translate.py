import requests
import json
import time
import logging
from typing import Callable, Iterable, Tuple, Dict

class Translator:
    def __init__(self, folder_id, token_getter:Callable, target_lang='en', max_batch_size=10000):
        self._logger = logging.getLogger('translator')
        self._folder_id = folder_id
        self._target_lang = target_lang
        self._token_getter = token_getter
        self._max_batch_size = max_batch_size

    def translate(self, text:str) -> str:
        """
        Translates text to target language. Total text length should be less than max_batch_size(default 10000).
        @param text: string to translate
        @return: translated string
        """
        if len(text) > self._max_batch_size:
            raise ValueError(f'text length should be less than {self._max_batch_size}')
        return self.translate_batch([text])['text'][0]
    
    def translate_batch(self, text_list:Iterable[str]) -> Dict[str, Iterable[str]]:
        """
        @param text_list: list of strings to translate
        @return: list of translated strings and list of detected languages
        """
        chunk = self._get_batches(text_list)
        translated = []
        detected_langs = []
        for i, batch in enumerate(chunk):
            total_len = sum(len(t) for t in batch)
            self._logger.info(f'translating chunk {i+1} of len {len(batch)} batches, total len {total_len}')
            translated_batch, detected_langs_batch = self._translate_batch(batch)
            translated.extend(translated_batch)
            detected_langs.extend(detected_langs_batch)
            self._logger.info(f'batch {i+1} translated')
        return {
            'text': translated,
            'lang': detected_langs
        }
    
    def _translate_batch(self, batch:Iterable[str]) -> Tuple[Iterable[str], Iterable[str]]:
        """
        @param batch: list of strings to translate, should be less than max_batch_size(default 10000) characters in total
        @return: list of translated strings and list of detected languages
        """
        batch = list(batch)
        symbols_count = sum(len(t) for t in batch)
        if symbols_count > self._max_batch_size:
            raise RuntimeError(f"batch is too big: {symbols_count}")
        body = {
            "targetLanguageCode": self._target_lang,
            "texts": batch,
            "folderId": self._folder_id,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer {0}".format(self._token_getter.iam_token),
        }
        while True:
            # retry until success
            try:
                resp = requests.post("https://translate.api.cloud.yandex.net/translate/v2/translate",
                    json=body,
                    headers=headers)
                resp.raise_for_status()
                resp = resp.json()
            except requests.HTTPError as e:
                self._logger.error(f'failed to translate: {e}')
            except json.JSONDecodeError as e:
                self._logger.error(f'failed to parse translate response: {e}')
            else:
                break
            time.sleep(1)
        return [t['text'] for t in resp['translations']], [t['detectedLanguageCode'] for t in resp['translations']]

    def _get_batches(self, big_batch:Iterable[str]) -> Iterable[Iterable[str]]:
        """
        The maximum total length of all strings in batch is max_batch_size(default 10000) characters.
        So we need to split the chank into batches.
        @param big_batch: list of strings to translate
        @return: list of batches(chunk)
        """
        sum_len = 0
        batch = []
        for text in big_batch:
            if sum_len + len(text) > self._max_batch_size:
                yield batch
                batch = []
                sum_len = 0
            batch.append(text)
            sum_len += len(text)
        if batch:
            yield batch
