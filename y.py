import time

current_direction = "F"
start_time = 0.0
alerted = False
stare_count = 0


def update2(new_direction):
    global current_direction, start_time, alerted, stare_count

    current_time = time.time()

    if new_direction != current_direction:
        old_direction = current_direction
        current_direction = new_direction
        
        print(f"direction changed from {old_direction} to {new_direction}")

        if new_direction == "F":
            start_time = 0.0
            alerted = False
        else:
            start_time = current_time
            alerted = False

    elif new_direction != "F" and start_time > 0:    #it means the direction does not change and we are not looking forward
        duration = current_time - start_time

        if duration >= 4.0 and not alerted:   #if we are looking in the same direction for more than 5 seconds and we have not alerted yet
            stare_count += 1
            alerted = True
        
            alert = {
                "type" : "stare_alert",
                "duration" : duration,
                "stare_count" :stare_count,
                "timestamp" : current_time
            }

            print(f"stare alert: {alert}")
    
    return None