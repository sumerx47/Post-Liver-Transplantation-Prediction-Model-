import psycopg2
import pandas as pd

# Establish a connection to the PostgreSQL database
conn = psycopg2.connect(
    host="127.0.0.1",
    port="5432",
    database="postgres",
    user="postgres",
    password="your password"
)

# Create a cursor object to interact with the database
cursor = conn.cursor()

try:
    # Roll back any previous transaction
    conn.rollback()

    # Execute the SQL query to retrieve the data from table df7
    cursor.execute("SELECT * FROM public.df7")

    # Fetch all the rows from the result
    rows = cursor.fetchall()

    # Get the column names from the cursor description
    column_names = [desc[0] for desc in cursor.description]

    # Create a pandas DataFrame from the rows and column names
    df = pd.DataFrame(rows, columns=column_names)

    # Print the DataFrame
    print(df)

    # Commit the transaction
    conn.commit()

except Exception as e:
    # Roll back the transaction if an error occurs
    conn.rollback()
    print("Error:", e)

finally:
    # Close the cursor and the connection
    cursor.close()
    conn.close()







