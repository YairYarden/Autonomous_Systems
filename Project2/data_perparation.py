def read_landmarks(filename):
    # Reads the world definition and returns a list of landmarks, our 'map'.
    # .
    # The returned dict contains a list of landmarks each with the
    # following information: {id, [x, y]}

    landmarks = []

    f = open(filename)

    for line in f:
        line_s = line.split('\n')
        line_spl = line_s[0].split(',')
        landmarks.append([float(line_spl[0]), float(line_spl[1])])

    return landmarks


def read_odometry(filename):
    # Reads the odometry and sensor readings from a file.
    #
    # The data is returned in a dict where the u_t and z_t are stored
    # together as follows:
    #
    # {odometry,sensor}
    #
    # where "odometry" has the fields r1, r2, t which contain the values of
    # the identically named motion model variables, and sensor is a list of
    # sensor readings with id, range, bearing as values.
    #
    # The odometry and sensor values are accessed as follows:
    # odometry_data = sensor_readings[timestep, 'odometry']
    # sensor_data = sensor_readings[timestep, 'sensor']

    sensor_readings = dict()

    first_time = True
    timestamp = 0
    f = open(filename)

    for line in f:

        line_s = line.split('\n')  # remove the new line character
        line_spl = line_s[0].split(' ')  # split the line

        if line_spl[0] == 'ODOMETRY':
            sensor_readings[timestamp] = {'r1': float(line_spl[1]), 't': float(line_spl[2]), 'r2': float(line_spl[3])}
            timestamp = timestamp + 1

    return sensor_readings