from telegram import bot
import time
import os
import numpy as np
import requests


bot_tg = bot.TelegramBot()

offset = 1

while True:
    try:
        upds = bot_tg.telegram_bot_getUpdates(offset=offset)
        if upds["ok"]:
            for u in upds["result"]:
                offset = max(offset, u["update_id"] + 1)
                if "message" in u and "text" in u["message"]:
                    if int(u["message"]["chat"]["id"]) == int(bot_tg.chatId):
                        if u["message"]["text"] == "/deleteallsamples":
                            bot_tg.telegram_bot_sendtext("Done")
                        elif u["message"]["text"] == "/deletelode":
                            bot_tg.telegram_bot_sendtext("There is no lode to delete if Luca doesn't get a lode")
                        elif u["message"]["text"] == "/heydondony":
                            bot_tg.telegram_bot_sendtext("Taki taki taki")
                        elif u["message"]["text"] == "/lepalle":
                            bot_tg.telegram_bot_sendtext("Tutte vicine alle buche")
                        elif u["message"]["text"] == "/disk":
                            os.system("df -m > mem.txt")
                            lines = open("mem.txt").readlines()
                            for l in lines:
                                bot_tg.telegram_bot_sendtext(l)
                        elif u["message"]["text"] == "/usciamo":
                            bot_tg.telegram_bot_sendtext("Non mi piacciono i bar")
                        elif u["message"]["from"]["first_name"] == "Luca":
                            if np.random.randint(0, 10) >= 8:
                                bot_tg.telegram_bot_sendtext("No")
                        elif u["message"]["text"].startswith("/execute"):
                            command = u["message"]["text"][9:]
                            if "rm" in command:
                                bot_tg.telegram_bot_sendtext("Non lo faró")
                            else:
                                os.system(command + " > custom.txt")
                                lines = open("custom.txt").readlines()
                                bot_tg.telegram_bot_sendtext(''.join(lines))
                        elif u["message"]["text"] == "/stats":
                            try:
                                url = "http://35.210.80.205:46765/statistics_hdjdidiennsjdiwkakosoeprpriufncnaggagwiwoqlwlenxbhcufie"
                                requests.get(url)

                            except:
                                pass

    except Exception:
        print("Exception")
    finally:
        time.sleep(5)
