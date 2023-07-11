#! /usr/bin/env python3

import os
import logging

import argparse
import itertools
import time
import traceback

import grpc

import recognition_pb2
import recognition_pb2_grpc

from modules.tokens import TokenUpdater
from modules.translate import Translator
from modules.speech_processor import SpeechProcessor
import pyaudio
import wave
import google.protobuf as pb

CHUNK_SIZE = 2048

path = os.path.dirname(os.path.abspath(__file__))
ca_path = os.path.join(path, 'russian_trusted_root_ca_pem.crt')

class SpeechRecognizer:
    recognition_options = {
        'audio_encoding': recognition_pb2.RecognitionOptions.PCM_S16LE,
        'sample_rate': 16000,
        'model': '',
        'language': 'ru-RU',
        'hypotheses_count': 0,
        'enable_profanity_filter': False,
        'enable_multi_utterance': True,
        'enable_partial_results': False,
        'no_speech_timeout': pb.duration_pb2.Duration(seconds=5),
        'max_speech_timeout': pb.duration_pb2.Duration(seconds=20),
        'hints': recognition_pb2.Hints(words=['лама', 'кубик', 'блок', 'башенка', 'цветные', 'тарелочки', 'тарелки',
                                              'кубики', 'тарелкам', 'блоки', 'башенку', 
                                              'тарелку', 'башенки', 'башенкам', 'по', 'тарелочкам'],
                                       eou_timeout=pb.duration_pb2.Duration(seconds=5),
                                       enable_letters=True),                                        
    }
    
    def __init__(self, token_getter, ca=None, chunk_size=CHUNK_SIZE):
        self._token_getter = token_getter
        self._ca = ca
        self._chunk_size = chunk_size
        self._logger = logging.getLogger('recognizer')
        self._logger.info('initializing microphone stream')
        self._open_mic_stream()
        
    
    def recognize(self, recognized_cb = None, normalized_result=True):
        ssl_cred = grpc.ssl_channel_credentials(
        root_certificates=open(self._ca, 'rb').read() if self._ca else None,)
        token_cred = grpc.access_token_call_credentials(self._token_getter.speech_token)
        channel = grpc.secure_channel(
            'smartspeech.sber.ru',
            grpc.composite_channel_credentials(ssl_cred, token_cred)
        )
        stub = recognition_pb2_grpc.SmartSpeechStub(channel)
        metadata_pairs = []
        con = stub.Recognize(itertools.chain(
            (recognition_pb2.RecognitionRequest(options=self.recognition_options),),
            self._chunks_generator(),
        ), metadata=metadata_pairs)
        try:
            for resp in con:
                
                # print(f'Got response: {resp}')
                if not resp.eou:
                    print('Got partial result:')
                    self._logger.info('got partial result:')
                if resp.eou_reason != recognition_pb2.EouReason.ORGANIC:
                    self._logger.info(f'End-of-utterance reason: {resp.eou_reason}')
                    continue
                for i, hyp in enumerate(resp.results):
                    text = hyp.normalized_text if normalized_result else hyp.text
                    self._logger.info(f'recognized text: {text}')
                    if recognized_cb and len(text) > 0:
                        recognized_cb(text)
        except grpc.RpcError as err:
            self._logger.error(f'RPC error: code = {err.code()}, details = {err.details()}')
        except Exception as e:
            self._logger.error(f'exception: {e}, {traceback.format_exc()}')
        else:
            self._logger.info('recognition has finished')
        finally:
            for m in con.initial_metadata():
                if m.key == 'x-request-id':
                    self._logger.info(f'RequestID: {m.value}')
            channel.close()

    def _open_mic_stream(self):
        self._pyaudio = pyaudio.PyAudio()
        self._stream = self._pyaudio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=self._chunk_size)

    def _chunks_generator(self):
        while True:
            data = self._stream.read(self._chunk_size, exception_on_overflow=False)
            yield recognition_pb2.RecognitionRequest(audio_chunk=data)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    speech_auth_token = os.environ['SPEECH_AUTH_TOKEN']
    oauth_token = os.environ['OAUTH_TOKEN']
    folder_id = os.environ['FOLDER_ID']
    token_updater = TokenUpdater(speech_auth_token, oauth_token)    
    translator = Translator(folder_id, token_updater)
    recognizer = SpeechRecognizer(token_updater, ca=ca_path)
    speech_processor = SpeechProcessor(translator)
    while True:
        try:
            recognizer.recognize(speech_processor.process, normalized_result=True)
        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.error(f'exception: {e}, {traceback.format_exc()}')
