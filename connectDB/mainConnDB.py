import mysql.connector

mydb = mysql.connector.connect(
  host="127.0.0.1",
  user="root",
  password="",
  database="restaurant"
)

print(mydb)
mycursor = mydb.cursor()

table_name = "test"
fields = ("name", "address", "text1", "text2", "text3")
values = ("John", "Highway21", "fuck", "dsd", "ssss")

# Construct placeholders for values
field_placeholders = ', '.join(['%s' for _ in fields])
print(field_placeholders)

# Construct the SQL query
sql = f"INSERT INTO {table_name} ({', '.join(fields)}) VALUES ({field_placeholders})"

# Execute the query
mycursor.execute(sql, values)

# Commit the transaction
mydb.commit()

print(mycursor.rowcount, "record inserted.")