from DKT.tiqu import run
import os
import json

storage = run()  # 获取DKT得到的知识状态
with open(os.path.join("data", "storage"), 'w', encoding='utf-8') as f:
    json.dump(storage, f, ensure_ascii=False)
