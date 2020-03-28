from telegram import bot
import time
import os

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
    except Exception:
        print("Exception")
    finally:
        time.sleep(5)
