import winsound

def trigger_alert(level):

    if level == "DROWSY":

        winsound.Beep(2500, 800)