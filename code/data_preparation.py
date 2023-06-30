import numpy as np

# ------------------- PART 1 - PARTICLE FILTER ------------------- #
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


def calculate_trajectory(trueOdometry):
    trueTrajectory = np.zeros((trueOdometry.__len__(), 3))

    for i in range(1, trueOdometry.__len__()):
        dr1 = trueOdometry[i - 1]['r1']
        dt = trueOdometry[i - 1]['t']
        dr2 = trueOdometry[i - 1]['r2']
        theta = trueTrajectory[i - 1, 2]
        dMotion = np.expand_dims(np.array([dt * np.cos(theta + dr1), dt * np.sin(theta + dr1), dr1 + dr2]), 0)
        trueTrajectory[i, :] = trueTrajectory[i - 1, :] + dMotion

    return trueTrajectory

def generate_measurement_odometry(trueOdometry, sigma_r1, sigma_t, sigma_r2):
    measurmentOdometry = dict()
    measured_trajectory = np.zeros((trueOdometry.__len__() + 1, 3))
    for i, timestamp in enumerate(range(trueOdometry.__len__())):
        dr1 = trueOdometry[timestamp]['r1'] + float(np.random.normal(0, sigma_r1, 1))
        dt = trueOdometry[timestamp]['t'] + float(np.random.normal(0,  sigma_t, 1))
        dr2 = trueOdometry[timestamp]['r2'] + float(np.random.normal(0,  sigma_r2, 1))
        measurmentOdometry[timestamp] = {'r1': dr1,
                                         't': dt,
                                         'r2': dr2}
        theta = measured_trajectory[i, 2]
        dMotion = np.expand_dims(np.array([dt*np.cos(theta + dr1), dt*np.sin(theta+ dr1),  dr1 + dr2]), 0)
        measured_trajectory[i + 1, :] = measured_trajectory[i, :] + dMotion

    return measured_trajectory

# ------------------- PART 2 - ICP ------------------- #
def get_pc(data, idx):
    return data.get_velo(idx)[:, :3]

def load_image(data,idx):
    ### Get the image data
    img_raw = data.get_cam2(idx) #TODO (hint- use get_cam2 in pykitti)
    return img_raw