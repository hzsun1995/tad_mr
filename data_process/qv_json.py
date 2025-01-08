import json

# 新的数据列表
new_data = []

# 逐行读取JSON文件
with open('/home/sunhaozhou/NumPro/data/highlight_val_release.jsonl', 'r') as file:
    for line in file:
        # 解析每一行的JSON对象
        item = json.loads(line)
        
        # 检查relevant_windows是否只有一个区间
        if len(item['relevant_windows']) == 1:
            # 提取所需字段
            new_item = {
                'id': f"{item['vid']}_{item['qid']}",
                'video': f"{item['vid']}.mp4",
                'start_time': item['relevant_windows'][0][0],
                'end_time': item['relevant_windows'][0][1],
                'query': item['query'],
                'duration': item['duration']
            }
            # 添加到新的数据列表
            new_data.append(new_item)

# 将新的数据写入JSON文件
with open('/home/sunhaozhou/NumPro/data/qv_new.json', 'w') as file:
    json.dump(new_data, file, indent=4)