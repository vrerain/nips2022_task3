import os
import json
from tkinter.messagebox import NO
from itsdangerous import exc
import pandas as pd
import numpy as np
from sqlalchemy import false
from tqdm import tqdm
from DKT.run import run
from datetime import datetime, timedelta
import time

storage = run()  # 获取DKT得到的知识状态
# with open(os.path.join("data", "storage"), 'w', encoding='utf-8') as f:
#     json.dump(storage, f, ensure_ascii=False)

with open(os.path.join("data", "storage"), 'w', encoding='utf-8') as f:
    json.dump({}, f, ensure_ascii=False)