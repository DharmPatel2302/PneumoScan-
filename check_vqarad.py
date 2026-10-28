import json
from collections import Counter

with open('89kps-osfstorage-archive/VQA_RAD Dataset Public.json') as f:
    data = json.load(f)

print('Total QA pairs:', len(data))
organs = Counter(d['image_organ'] for d in data)
print('By organ:', dict(organs))
answer_types = Counter(d['answer_type'] for d in data)
print('By answer type:', dict(answer_types))
q_types = Counter(d['question_type'] for d in data)
print('By question type:', dict(q_types))
imgs = set(d['image_name'] for d in data)
print('Unique images:', len(imgs))

chest = [d for d in data if d['image_organ'] == 'CHEST']
print('Chest QA pairs:', len(chest))
chest_open = [d for d in chest if d['answer_type'] == 'OPEN']
print('Chest OPEN pairs:', len(chest_open))

print('\nSample CHEST questions:')
for q in chest[:5]:
    print(f"  [{q['answer_type']}] Q: {q['question']}  A: {q['answer']}")
