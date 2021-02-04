import smtplib, ssl
import os


def send_email(message, to_addrs="jljh1g18@soton.ac.uk"):
    """ Send a plain text email """
    with smtplib.SMTP_SSL(
        "smtp.gmail.com", 
        465, 
        context=ssl.create_default_context()
    ) as smtp_server:
        user = "ml.fakenews.notif@gmail.com"
        passw = os.getenv("ML_FAKENEWS_NOTIF")
        smtp_server.login(user, passw)
        
        smtp_server.sendmail(user, to_addrs, "\n"+message)
    