#! /usr/bin/env python3

import requests
import json
import os
import uuid
import datetime
import time
import logging
import threading
from dateutil import parser
import traceback

class TokenUpdater:
    def __init__(self, speech_auth_token, oauth_token):
        self._logger = logging.getLogger('token_updater')
        self._speech_token = None
        self._iam_token = None
        self._speech_auth_token = speech_auth_token
        self._oauth_token = oauth_token
        self._speech_token_expires = datetime.time()
        self._iam_token_expires = datetime.time()
        self._lock = threading.Lock()
        self._update_speech_token()
        self._update_iam_token()
        self._thread = threading.Thread(target=self._expire_watcher, daemon=True, name='token_updater')
        self._thread.start()

    def _expire_watcher(self):
        while True:
            try:
                self._logger.debug(f'speech token time left: {self._speech_token_expires - datetime.datetime.now()}')
                self._logger.debug(f'iam token time left: {self._iam_token_expires - datetime.datetime.now()}')
                if self._speech_token_expires - datetime.timedelta(minutes=20) <= datetime.datetime.now():
                    self._logger.info('speech token expires soon')
                    self._update_speech_token()
                if self._iam_token_expires - datetime.timedelta(hours=6) <= datetime.datetime.now():
                    self._logger.info('iam token expires soon')
                    self._update_iam_token()
            except Exception as e:
                self._logger.error(f'error in token update thread: {e}')
                self._logger.error(traceback.format_exc())
            time.sleep(10)

    def _update_iam_token(self):
        with self._lock:
            while True:
                try:
                    self._logger.info('updating iam token')
                    resp = requests.post('https://iam.api.cloud.yandex.net/iam/v1/tokens',
                        data=json.dumps({'yandexPassportOauthToken': self._oauth_token}))
                    resp.raise_for_status()
                    resp = resp.json()
                except requests.HTTPError as e:
                    self._logger.error(f'failed to update token: {e}')
                except json.JSONDecodeError as e:
                    self._logger.error(f'failed to parse token response: {e}')
                else:
                    break
                time.sleep(1)
            parsed = parser.parse(resp['expiresAt'])
            self._iam_token_expires = datetime.datetime.fromtimestamp(parsed.timestamp())
            self._iam_token = resp['iamToken']
            self._logger.info(f'updated iam token')

    def _update_speech_token(self):
        with self._lock:
            while True:
                try:
                    self._logger.info('updating speech token')
                    resp = requests.post('https://ngw.devices.sberbank.ru:9443/api/v2/oauth', 
                    headers={'Authorization' : f"Basic {self._speech_auth_token}",
                            'RqUID': str(uuid.uuid4()),
                            'Content-Type': 'application/x-www-form-urlencoded'},
                        data={'scope': 'SALUTE_SPEECH_CORP'})
                    resp.raise_for_status()
                    resp = resp.json()
                except requests.HTTPError as e:
                    self._logger.error(f'failed to update token: {e}')
                except json.JSONDecodeError as e:
                    self._logger.error(f'failed to parse token response: {e}')
                else:
                    break
                time.sleep(1)
            ms = int(resp['expires_at'])
            self._speech_token_expires = datetime.datetime.fromtimestamp(ms / 1000)
            self._speech_token = resp['access_token']
            self._logger.info(f'updated speech token')
            
    @property
    def speech_token(self):
        with self._lock:
            return self._speech_token
        
    @property
    def iam_token(self):
        with self._lock:
            return self._iam_token
        
class SpeechRecognizer:
    def __init__(self, token_getter):
        self._logger = logging.getLogger('visper')
        self._token_getter = token_getter

    def _recognize(self, audio):
        pass

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('token_updater').setLevel(logging.DEBUG)
    speech_auth_token = os.environ['SPEECH_AUTH_TOKEN']
    oauth_token = os.environ['OAUTH_TOKEN']
    folder_id = os.environ['FOLDER_ID']
    token_updater = TokenUpdater(speech_auth_token, oauth_token)
    speech_recongnizer = SpeechRecognizer(token_updater)
    try:
        while True:
            print(token_updater.speech_token)
            print(token_updater.iam_token)
            time.sleep(60)
    except KeyboardInterrupt:
        pass