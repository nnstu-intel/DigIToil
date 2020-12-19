import smtplib                                              
import os                                                   
import mimetypes                                            
from email import encoders                                  
from email.mime.base import MIMEBase                        
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fpdf import FPDF
import time
#Функция отправки сообщения
def send_email(addr_to, msg_subj, msg_text, files):
    addr_from = "testing.python@mail.ru"                         # Отправитель
    password  = "dishiestduke"                                  # Пароль

    msg = MIMEMultipart()                                   # Создаем сообщение
    msg['From']    = addr_from                              
    msg['To']      = addr_to                                
    msg['Subject'] = msg_subj                               

    body = msg_text                                         # Текст сообщения
    msg.attach(MIMEText(body, 'plain'))                     # Добавляем в сообщение текст

    process_attachement(msg, files)
    server=smtplib.SMTP('smtp.mail.ru',25) # это не трогать!!! работает ток с mail.ru
    server.starttls()
    server.login(addr_from,password)
    server.send_message(msg)
    server.quit()
# Функция по обработке списка, добавляемых к сообщению файлов  
def process_attachement(msg, files):                     
    for f in files:
        if os.path.isfile(f):
            attach_file(msg,f)
        elif os.path.exists(f):                             
            dir = os.listdir(f)
            for file in dir:                                
                attach_file(msg,f+"/"+file)                 
# Функция по добавлению конкретного файла к сообщению
def attach_file(msg, filepath):                             
    filename = os.path.basename(filepath)                  
    ctype, encoding = mimetypes.guess_type(filepath)        
    if ctype is None or encoding is not None:               
        ctype = 'application/octet-stream'                  
    maintype, subtype = ctype.split('/', 1)     
    with open(filepath, 'rb') as fp:
            file = MIMEBase(maintype, subtype)              
            file.set_payload(fp.read())                     
            fp.close()
            encoders.encode_base64(file)                    
    file.add_header('Content-Disposition', 'attachment', filename=filename) 
    msg.attach(file)       
def pdf_write(image):         
   pdf = FPDF()
   pdf.add_page()
   pdf.set_font("Arial", size=12)
   center="Violation report"
   pdf.cell(200, 10, txt=center, ln=1, align="C")
   pdf.image(image, x=10, y=20, w=100)
   pdf.ln(85)  # ниже на 85
   name = 'Lack of a helmet or building vest'
   pdf.cell(200, 10, txt=name, ln=1)
   vremy=str(time.asctime())
   pdf.cell(200,10,txt=vremy,ln=1)
   surname='Responsible: Sidorov P.A.'
   pdf.cell(200,10,txt=surname,ln=1)
   city='Nizhny Novgorod,Minin Street'
   pdf.cell(200,10,txt=city,ln=1)
   pdf.set_line_width(1)
   pdf.set_draw_color(0, 0, 0)
   pdf.line(20, 115, 100, 115)
   pdf.output("Output/pdf/Accountability.pdf")
                                                                   