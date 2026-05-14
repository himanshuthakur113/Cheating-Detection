import time

current_direction = "F"
start_time = 0
look_cnt = 0

def update(new_direction):
    global current_direction, start_time, look_cnt

    current_time = time.time()

    if new_direction == current_direction:
        return
    
    old_direction = current_direction
    current_direction = new_direction

    print(f"Direction changed from {old_direction} to {new_direction}")

    duration = current_time - start_time
    print(duration)

    start_time = current_time

    if old_direction != "F":
        if duration  > 1:
            look_cnt += 1
    
    print(f"look count: {look_cnt}")
