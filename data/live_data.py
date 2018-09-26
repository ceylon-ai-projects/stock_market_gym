import os
import threading

dirname = os.path.dirname(__file__)


def get_live_data(pair_name, call_back, interval="1T"):
    import datetime
    import json

    import pandas as pd
    import websocket

    file_path = "./live/{pair}/{pair}{interval}.csv".format(
        pair=pair_name, interval=interval)
    file_path = os.path.join(dirname, file_path)
    direcotry_path = os.path.join(dirname, "./live/{}".format(pair_name))
    if os.path.exists(direcotry_path) is False:
        os.makedirs(direcotry_path)

    def on_open(ws):
        # json_data = json.dumps({'ticks': 'frxEURUSD'})
        if interval == "1T":
            json_data = json.dumps({
                "ticks_history": "frx{}".format(pair_name),
                "end": "latest",
                "start": 1,
                "style": "ticks",
                "adjust_start_time": 1,
                "count": 5000
            })
            ws.send(json_data)
        elif interval == "1M":
            json_data = json.dumps({
                "ticks_history": "frxEURUSD",
                "end": "latest",
                "start": 2,
                "granularity": 60,
                "style": "candles",
                "adjust_start_time": 1,
                "count": 5000
            })
            ws.send(json_data)

    def on_message_candle(ws, message):
        try:
            message = json.loads(message)
            candles = message['candles']
            history = []
            for i in range(len(candles)):
                candle = candles[i]
                time = datetime.datetime.fromtimestamp(int(float(candle['epoch']))).strftime('%Y-%m-%d %H:%M:%S')
                open = float(candle['open'])
                close = float(candle['close'])
                high = float(candle['high'])
                low = float(candle['low'])
                # int(
                #     float(times[i]))  #

                history.append([time, open, high, low, close])

            history = pd.DataFrame(history)
            history.columns = ['Date_Time', "Open", "High", "Low", "Close"]
            history.to_csv(file_path, mode="w")

            # print(history.tail())
            ws.keep_running = False
            # np.savetxt("EURUSD1T.csv", history, delimiter=",")
            call_back(history)
        except Exception as e:
            print(e)

    def on_message_ticks(ws, message):
        try:
            message = json.loads(message)
            # print(message)
            prices = message['history']['prices']
            times = message['history']['times']
            history = []
            for i in range(len(prices)):
                # int(
                #     float(times[i]))  #
                time = datetime.datetime.fromtimestamp(int(float(times[i]))).strftime('%Y-%m-%d %H:%M:%S')
                history.append([time, float(prices[i])])
            history = pd.DataFrame(history)
            history.columns = ['Date_Time', 'Price']
            # parse_dates = ['Date_Time']

            history.to_csv(file_path, mode="w")

            # print(history.tail())
            ws.keep_running = False
            # np.savetxt("EURUSD1T.csv", history, delimiter=",")
            call_back(history)
        except Exception as e:
            print(e)

    apiUrl = "wss://ws.binaryws.com/websockets/v3?app_id=####"
    ws = websocket.WebSocketApp(apiUrl,
                                on_message=on_message_candle,
                                on_open=on_open)

    while True:
        try:
            ws.run_forever()
        except:
            pass
    # # ws.keep_running =
    # ws.run_forever()
    # # wst = threading.Thread(target=ws.run_forever)
    # # wst.daemon = True
    # # wst.start()
    # # wst.join()
    return file_path
