import logging
from modules.translate import Translator
import threading
import requests

synonyms = {
    'blocks': [
        'colored blocks',
        'colored cubes',
        'colored bricks',
        'cubes',
        'bricks',
    ],
    'block': [
        'colored block',
        'colored cube',
        'colored brick',
        'cube',
        'brick',
    ],
    'towers': [
        'turrets',
        'spires',
        'minarets',
        'obelisks',
        'steeples',
        'peaks',
        'cupolas',
        'pinnacles',
        'columns',
        'pillars',
        'masts'
    ],
    'tower': [
        'turret',
        'spire',
        'minaret',
        'obelisk',
        'steeple',
        'peak',
        'cupola',
        'pinnacle',
        'column',
        'pillar',
        'mast'
    ],
    'plates': [
        'colored plates',
        'dishes',
        'saucers',
        'trays',
        'platters',
        'salvers',
        'chargers',
        'cups',
        'bowls',
        'vessels',
        'containers',
        'pans',
        'crocks',
        'pots',
        'jars',
        'jugs',
        'piles'
    ],
    'plate': [
        'colored plate',
        'dish',
        'saucer',
        'tray',
        'platter',
        'salver',
        'charger',
        'cup',
        'bowl',
        'vessel',
        'container',
        'pan',
        'crock',
        'pot', 
        'jar',
        'jug',
        'pile'
    ],
    'fishes': [
        'small fishes',
        'herrings',
        'crucian carps'
        'salmons'
    ],
    'fish': [
        'small fish',
        'herring',
        'crucian carp'
        'salmon'
    ],
    "": [
        'please ',
        'please, ',
        ',please',
        ' please'
    ]
}

class SpeechProcessor:
    def __init__(self, translator:Translator):
        self._logger = logging.getLogger('speech_processor')
        self._translator = translator
        self._request_thread = None

    def process(self, text):
        text, is_appeal = self._remove_appeal(text)
        text = self._crop_execute(text)
        if len(text) < 10:
            return 'TOO_SHORT'
        rus = text
        text = self._translator.translate(text)
        text = text.lower()
        text = self._rephrase_eng(text)
        text = self._remove_dot_at_the_end(text)
        self._logger.info(f'is_appeal: {is_appeal}, processed: {text}')
        if is_appeal:
            ret = self._request_execution(text, rus)
            if ret:
                return 'OK'
            else:
                return 'BUSY'

    def _request_execution(self, text, rus):
        def post_request():
            try:
                req = requests.post('http://0.0.0.0:5000/execute', json={'task': text, "rus": rus})
            except Exception as e:
                self._logger.error(f'failed to post request: {e}')
            else:
                self._logger.info(f'request posted: {req.status_code}, {req.text}')

        if self._request_thread is not None:
            if self._request_thread.is_alive():
                return False
        self._request_thread = threading.Thread(target=post_request, daemon=True, name='request_thread')
        self._request_thread.start()
        return True
    

    def _rephrase_eng(self, text):
        global synonyms
        for original, syns in synonyms.items():
            for synonym in syns:
                text = text.replace(synonym, original)
        return text

    def _remove_appeal(self, text):
        is_appeal = False
        appeals = ['Лама', 'Лама,', 'Лама.', 'Лама!', 'Лама?', 'Лама:', 'Лама;', 'Лама-']
        appeals.extend(['Мама', 'Мама,', 'Мама.', 'Мама!', 'Мама?', 'Мама:', 'Мама;', 'Мама-'])
        appeals.extend(['Плава', 'Плава,', 'Плава.', 'Плава!', 'Плава?', 'Плава:', 'Плава;', 'Плава-'])
        appeals.extend(['Ламу', 'Ламу,', 'Ламу.', 'Ламу!', 'Ламу?', 'Ламу:', 'Ламу;', 'Ламу-'])
        lowered = []
        for appeal in appeals:
            lowered.append(appeal.lower())
        appeals.extend(lowered)
        appeals.sort(key=len, reverse=True)
        for appeal in appeals:
            does_exist = appeal in text
            if does_exist:
                text = text.split(appeal)[1]
                is_appeal = True
        text = text.strip()
        return text, is_appeal

    def _crop_execute(self, text):
        phrase = ['выполни', 'выполнить', 'выполнишь', 'выполните', 'выполняй', 'выполнять', 'выполняешь', 'выполняете']
        for word in phrase:
            if word in text:
                text = text.split(word)[0]
                break
        return text

    def _remove_dot_at_the_end(self, text):
        if text.endswith('.'):
            text = text[:-1]
        return text
