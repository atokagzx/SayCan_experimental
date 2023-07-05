import requests
import json
import time
import logging
from typing import Callable

class Translator:
    def __init__(self, folder_id, token_getter:Callable, target_lang='en'):
        self._logger = logging.getLogger('translator')
        self._folder_id = folder_id
        self._target_lang = target_lang
        self._token_getter = token_getter

    def translate(self, text:str) -> str:
        body = {
            "targetLanguageCode": self._target_lang,
            "texts": [text],
            "folderId": self._folder_id,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer {0}".format(self._token_getter.iam_token),
        }

        while True:
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
        return resp['translations'][0]['text']