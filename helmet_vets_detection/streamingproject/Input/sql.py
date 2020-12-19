import psycopg2
from fpdf import FPDF
import time

conn = psycopg2.connect(dbname='build', user='postgres', 
                        password='All0887', host='127.0.0.1',port='5432')


def input_sql(h):
    cur = conn.cursor()
    cur.execute(("insert into build values( bytea('"+h+"'), 'Sidorov','Attention, a person without a uniform','"+str(time.asctime())+"');"))
    conn.commit()
    cur.close()
    
def print_sql():
    cur = conn.cursor()
    cur.execute("SELECT * from build")
  
    rows = cur.fetchall()
    for row in rows:  
        print("id =", row[0].tobytes().strip().decode( "utf-8" ))
        print("NAME =", row[1])
        print("image =", row[2])
        u=row[0].tobytes().strip().decode( "utf-8" )
        print(u)
    cur.close()  
    return u

print("Operation done successfully")  
  
def pdf_write(u):
            
   pdf = FPDF()
   pdf.add_page()
   pdf.set_font("Arial", size=12)
   center="Violation report"
   pdf.cell(200, 10, txt=center, ln=1, align="C")
   pdf.image(u, x=10, y=20, w=100)
   pdf.ln(85)  # ниже на 85
   name = 'Lack of a helmet or building vest'
   pdf.cell(200, 10, txt=name, ln=1)
   
   surname='Responsible: Sidorov P.A.'
   pdf.cell(200,10,txt=surname,ln=1)
   city='Nizhny Novgorod,Minin Street'
   pdf.cell(200,10,txt=city,ln=1)
   pdf.set_line_width(1)
   pdf.set_draw_color(0, 0, 0)
   pdf.line(20, 115, 100, 115)
   pdf.output("naruh.pdf")                                 