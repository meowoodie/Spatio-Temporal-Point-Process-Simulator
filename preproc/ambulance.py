import arrow
import numpy as np


days     = 1
morethan = 20

with open("data/eventos_desagregados.csv", "r") as f:
    data = []
    for line in f.readlines()[1:]:
        rawdata        = line.strip().split(",")
        time, lat, lng = rawdata[1], rawdata[4], rawdata[5]
        if time.strip() != "" and lat.strip() != "" and lng.strip() != "":
            date, time = time.strip().split(" ")
            mon, date, year = date.strip().split("/")
            hour, mins      = time.strip().split(":")
            year, mon, date, hour, mins = int("20" + year), int(mon), int(date), int(hour), int(mins)
            timestamp       = float(arrow.get(year, mon, date, hour, mins, 0).timestamp)
            lat = float(lat)
            lng = float(lng)
            if lat < -22.621 and lat > -23.226 and lng < -43.050 and lng > -43.868:
                data.append([timestamp, lat, lng])

    data = np.array(data)
    # reorder the dataset by timestamps
    order = data[:, 0].argsort()
    data  = data[order]
    print(data.shape)

    # partition the original sequence
    stack_list = []
    last_i     = 0
    max_len    = 0
    for i in range(len(data)):
        if data[i, 0] - data[last_i, 0] > days * 24 * 3600:
            seq = data[last_i:i, :]
            seq[:, 0] -= seq[0, 0]
            max_len = len(seq) if len(seq) > max_len else max_len
            if len(seq) >= morethan:
                stack_list.append(seq)
            last_i = i
    new_data = np.zeros((len(stack_list), max_len, 3))
    for j in range(len(stack_list)):
        new_data[j, :len(stack_list[j]), :] = stack_list[j]
    print(new_data.shape)

    expert_seq = new_data
    T         = expert_seq[:, :, 0].max() # days * 24 * 3600
    n_seq     = expert_seq.shape[0]
    step_size = expert_seq.shape[1]
    x_nonzero = expert_seq[:, :, 1][np.nonzero(expert_seq[:, :, 1])]
    y_nonzero = expert_seq[:, :, 2][np.nonzero(expert_seq[:, :, 2])]
    xlim      = [ x_nonzero.min(), x_nonzero.max() ]
    ylim      = [ y_nonzero.min(), y_nonzero.max() ]

    np.save('data/rescale.ambulance.perday.npy', expert_seq)

    print(expert_seq.shape)
    print("T", T)
    print("n_seq", n_seq)
    print("step_size", step_size)
    print("xlim", xlim)
    print("ylim", ylim)
    print(expert_seq[1, :, :])
    

