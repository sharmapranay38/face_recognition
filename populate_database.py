import os
import psycopg2
from imgbeddings import imgbeddings
from PIL import Image

# Database connection details
DB_CONNECTION_STRING = "postgres://avnadmin:AVNS_JehIaMQy7ho7CrQGG0W@pg-396e8f1c-sharmapranay38-f5ed.i.aivencloud.com:16167/defaultdb?sslmode=require"

# Specify the path to the stored faces
STORED_FACES_DIR = "C://Users//LENOVO//Desktop//mait hackathon//new//stored-faces//"

def populate_database():
    # Connect to the database
    conn = psycopg2.connect(DB_CONNECTION_STRING)
    cur = conn.cursor()

    # Drop the pictures table if it exists
    cur.execute("DROP TABLE IF EXISTS pictures;")

    # Create the pictures table
    cur.execute("""
        CREATE TABLE pictures (
            picture TEXT PRIMARY KEY,
            embedding FLOAT8[]
        );
    """)

    # Load the imgbeddings model
    ibed = imgbeddings()

    # Iterate through the images in the stored faces directory
    for filename in os.listdir(STORED_FACES_DIR):
        filepath = os.path.join(STORED_FACES_DIR, filename)
        
        # Check if it's a file
        if os.path.isfile(filepath):
            # Open the image and calculate embeddings
            img = Image.open(filepath)
            embedding = ibed.to_embeddings(img)
            
            # Insert the filename and embedding into the database
            cur.execute("INSERT INTO pictures (picture, embedding) VALUES (%s, %s)", (filename, embedding[0].tolist()))
            print(f"Inserted: {filename}")

    # Commit the changes and close the connection
    conn.commit()
    cur.close()
    conn.close()
    print("Database populated successfully.")

if __name__ == "__main__":
    populate_database()
