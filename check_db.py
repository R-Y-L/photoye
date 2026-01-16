#!/usr/bin/env python3
"""临时脚本：检查数据库状态"""

from database import get_connection
import os

conn = get_connection()
cursor = conn.cursor()

# 查看所有照片的分类
print("=== 数据库中的照片分类 ===")
cursor.execute('SELECT id, filepath, category, status FROM photos ORDER BY filepath')
rows = cursor.fetchall()
for row in rows:
    filename = os.path.basename(row['filepath'])
    print(f"  {filename}: category={row['category']}, status={row['status']}")

# 查看所有人脸
print("\n=== 数据库中的人脸 ===")
cursor.execute('''
    SELECT f.id, f.photo_id, f.person_id, f.confidence, p.filepath
    FROM faces f
    JOIN photos p ON p.id = f.photo_id
    ORDER BY f.photo_id, f.id
''')
faces = cursor.fetchall()
for face in faces:
    filename = os.path.basename(face['filepath'])
    print(f"  Face {face['id']}: photo={filename}, person_id={face['person_id']}, conf={face['confidence']:.3f}")

# 查看所有人物
print("\n=== 数据库中的人物 ===")
cursor.execute('SELECT id, name FROM persons ORDER BY id')
persons = cursor.fetchall()
for person in persons:
    print(f"  Person {person['id']}: {person['name']}")

conn.close()
