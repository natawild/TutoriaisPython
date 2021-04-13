
import pandas as pd 
import numpy as np 
from threading import Timer 
from datetime import datetime
import schedule 
import time 
import speedtest




j = speedtest
j.get_best_server()
j.download()
j.upload()
res = j.results.dict()
print(res["download"], res["upload"], res["ping"])


"""
def internet():
	df = pd.read_excel('dadosInternet.xlsx',engine='openpyxl')
	s = speedtest.Speedtest()
	data_atual = datetime.now().strftime('%d/%m/%Y')
	hora_atual = datetime.now().strftime('%H:%M')
	velocidade = s.download(threads=None)*(10**-6)
	df.loc[len(df)] = [data_atual, hora_atual, velocidade]
	df.to_excel('dadosInternet.xlsx',index=False)
	Timer(1800, internet).start()


#internet()

"""

