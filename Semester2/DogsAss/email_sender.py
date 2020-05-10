import smtplib 
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from email.mime.base import MIMEBase 
from email import encoders 
import os.path
  
def send_email(subject, message, attachment_path):
    #IF NO ATTACHMENT DONT SEND EMAIL
    if not os.path.isfile(attachment_path):
        return
    msg = MIMEMultipart() 
    msg['From'] = "sochgreg@gmail.com"
    msg['To'] = "19765251@students.lincoln.ac.uk"
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain')) 
    attachment = open(attachment_path, "rb")
    p = MIMEBase('application', 'octet-stream') 
    p.set_payload((attachment).read()) 
    encoders.encode_base64(p) 
    p.add_header('Content-Disposition', "attachment; filename= %s" % attachment_path) 
    msg.attach(p) 
    s = smtplib.SMTP('smtp.gmail.com', 587) 
    s.starttls() 
    s.login("sochgreg@gmail.com", "pass100word") 
    message = msg.as_string()
    s.sendmail("sochgreg@gmail.com", "19765251@students.lincoln.ac.uk", message) 
    s.quit() 
    
