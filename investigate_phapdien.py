from datasets import load_dataset

ds = load_dataset('tmquan/phapdien-moj-gov-vn', 'articles', split='train[:1]')
row = ds[0]

print('=== ARTICLES KEYS ===')
print(list(row.keys()))
print(f'\nTotal fields: {len(row.keys())}')

print('\n=== SAMPLE ARTICLE TITLE ===')
print(row.get('article_title', 'N/A'))

print('\n=== TEXT FIELDS CHECK ===')
for field in ['text', 'markdown', 'noi_dung', 'content', 'article_text', 'article_content']:
    if field in row:
        val = row[field]
        print(f'{field}: EXISTS ({len(str(val))} chars)')
    else:
        print(f'{field}: NOT FOUND')

print('\n=== FIRST 200 CHARS OF LONGEST TEXT FIELD ===')
for k, v in row.items():
    if isinstance(v, str) and len(v) > 50:
        print(f'{k}: {v[:200]}...')
        break
