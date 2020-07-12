import requests
from requests.exceptions import *

DONDONYBot_TOKEN = "1110199189:AAEeWiXJuDVKFdcaDX_pWQN2ntU4ALwX8MI"

CHATID = "-483064292"


class TelegramBot:

    def __init__(self, botid=DONDONYBot_TOKEN, chatId=CHATID):
        self.id = botid
        self.chatId = CHATID

    def telegram_bot_sendtext(self, bot_message):

        try:
            send_text = 'https://api.telegram.org/bot' + self.id + '/sendMessage?chat_id=' + self.chatId + '&parse_mode=Markdown&text=' + bot_message

            response = requests.get(send_text)

            return response.json()

        except RequestException:
            pass

    def telegram_bot_getUpdates(self):
        send_text = 'https://api.telegram.org/bot' + self.id + '/getUpdates'

        response = requests.get(send_text)

        return response.json()
