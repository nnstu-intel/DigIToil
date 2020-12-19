import psycopg2
from fpdf import FPDF
import datetime
#bname=input()
#polz=input()
#passw=input()
#hos=input()
#por=input()
conn = psycopg2.connect(dbname='build', user='postgres', 
                        password='dishiestduke628', host='127.0.0.1',port='5432')
#name=input()
#surname=input()
#day=str(datetime.date.today())
#city=input()
#mail=input()
#phone=input()
#pos=input()
row=[]
#path='C:/Users/Mvideo/Desktop/С„РѕС‚Рѕ/25.jpg'
def input_sql(n,s,d,c,e,p,ps):
    cur = conn.cursor()
    cur.execute("insert into employees(first_name,second_name,birthday,town,email,phone,positions) values('"+n+"','"+s+"','"+d+"','"+c+"','"+e+"','"+p+"','"+ps+"');")
    conn.commit()
    cur.close()
    
def print_sql():
    cur = conn.cursor()
    cur.execute("SELECT * from employees")
  
    rows = cur.fetchall()
    for row in rows:  
        print("id =", row[0])
        print("NAME =", row[1])
#        print("image =", row[2].tobytes().strip().decode( "utf-8" ))
#        u=row[2].tobytes().strip().decode( "utf-8" )
       # print(u)
    cur.close()  
    return rows

print("Operation done successfully")  
  
def pdf_write(u):
            
   pdf = FPDF()
   pdf.add_page()
   pdf.set_font("Arial", size=12)
   center="Violation report"
   pdf.cell(200, 10, txt=center, ln=1, align="C")
   pdf.image(u, x=10, y=20, w=100)
   pdf.ln(85)  
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
   
def naruh_sql(n,s,naruh):
    cur = conn.cursor()
    cur.execute("SELECT * from employees where first_name='"+n+"' and second_name='"+s+"';" )
    
    rows = cur.fetchall()
    for row in rows:  
        print("id =", row[0])
        print("NAME =", row[1])
        
    con = conn.cursor()
    con.execute("insert into nonobservance(first_name,second_name,violation,email,phone,positions,trespassing) values('"+row[0]+"','"+row[1]+"','"+str(datetime.date.today())+"','"+row[4]+"','"+row[5]+"','"+row[6]+"','"+naruh+"');")
    conn.commit()
    con.close()
    cur.close()  
    
#new_sql('Иван','Иванов')    
#input_sql(name,surname,day,city,mail,phone,pos)   
#g=print_sql()   
#pdf_write(g)                                   