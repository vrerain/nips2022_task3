import os
import json
from DKT.run import run

storage = run()  # 获取DKT得到的知识状态
# with open(os.path.join("data", "storage"), 'w', encoding='utf-8') as f:
#     json.dump(storage, f, ensure_ascii=False)

with open(os.path.join("data", "storage"), 'w', encoding='utf-8') as f:
    json.dump({}, f, ensure_ascii=False)
