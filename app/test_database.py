# import sqlite3

# conn = sqlite3.connect("/home/debjit/spacy/image_search_project_v2/data/metadata.db")
# cursor = conn.cursor()

# # Find bad rows
# cursor.execute("SELECT id, filename FROM image_metadata WHERE filename = 'path'")
# bad_rows = cursor.fetchall()
# print(f"Found {len(bad_rows)} bad rows")

# # Delete them
# cursor.execute("DELETE FROM image_metadata WHERE filename = 'path'")
# conn.commit()
# conn.close()


import sqlite3

conn = sqlite3.connect("/home/debjit/spacy/image_search_project_v2/data/metadata.db")
cursor = conn.cursor()

cursor.execute("SELECT id, filename FROM image_metadata")
rows = cursor.fetchall()

print(f"Total rows: {len(rows)}")
for row in rows:
    _id, fname = row
    print(f"[{_id}] {fname}")
    if not fname or fname == "path":
        print(f"❌ Broken: ID {_id} -> {fname}")

conn.close()