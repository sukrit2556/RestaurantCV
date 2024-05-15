
import mysql.connector
def connect_db():
    mydb = mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="",
        database="restaurant"
        )
    return mydb


def insert_db(table_name, field_list, value_list, verbose = False):
    mydb = connect_db()
    mycursor = mydb.cursor()

    # Construct placeholders for values
    field_placeholders = ', '.join(['%s' for _ in field_list])

    # Construct the SQL query
    sql = f"INSERT INTO {table_name} ({', '.join(field_list)}) VALUES ({field_placeholders})"
    if verbose == True:
        print("Generated SQL query:", sql)

    # Execute the query
    mycursor.execute(sql, value_list)

    # Commit the transaction
    mydb.commit()

    print(mycursor.rowcount, "record inserted.")
    mycursor.close()
    mydb.close()

def update_db(table_name, field_to_edit, new_value, condition_list, verbose = False):
    mydb = connect_db()
    mycursor = mydb.cursor()

    # Create placeholders for conditions
    condition_placeholders = ' AND '.join(['{}'.format(condition) for condition in condition_list])
    print(condition_placeholders)

    # Construct the SQL query string with placeholders
    sql = f"UPDATE {table_name} SET {field_to_edit} = %s WHERE {condition_placeholders}"


    # Print the SQL query (for debugging purposes)
    if verbose == True:
        print("Generated SQL query:", sql)

    # Execute the query
    mycursor.execute(sql, [new_value])

    # Commit the transaction
    mydb.commit()

    print(mycursor.rowcount, "record(s) updated.")
    mycursor.close()
    mydb.close()

def delete_data_db(table_name, condition_list, verbose = False):
    mydb = connect_db()
    mycursor = mydb.cursor()

    # Create placeholders for conditions
    condition_placeholders = ' AND '.join(['{}'.format(condition) for condition in condition_list])
    print(condition_placeholders)

    # Construct the SQL query string with placeholders
    sql = f"DELETE FROM {table_name} WHERE {condition_placeholders}"


    # Print the SQL query (for debugging purposes)
    if verbose == True:
        print("Generated SQL query:", sql)

    # Execute the query
    mycursor.execute(sql)

    # Commit the transaction
    mydb.commit()

    print(mycursor.rowcount, "record(s) deleted.")
    mycursor.close()
    mydb.close()

def select_db(table_name, field_name:list, where_condition:list, verbose = False):
    mydb = connect_db()
    mycursor = mydb.cursor()

    formated_field_name = ', '.join(['{}'.format(field_names) for field_names in field_name])
    formated_condition = ' AND '.join(['{}'.format(where_conditions) for where_conditions in where_condition])
    # Construct the SQL query string with placeholders
    sql = f"SELECT {formated_field_name} from {table_name} WHERE {formated_condition}"
    if verbose == True:
        print("Generated SQL query:", sql)

    mycursor.execute(sql)

    myresult = mycursor.fetchall()
    
    mycursor.close()
    mydb.close()

    return sql, myresult # for subquery uses
