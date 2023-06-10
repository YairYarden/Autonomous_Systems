import numpy as np
import graphs

class ParticlesFilter:
    def __init__(self, worldLandmarks, sigma_r1, sigma_t, sigma_r2, sigma_range, sigma_bearing, numberOfParticles=500):
        """
        Initialization of the particle filter
        """

        # Initialize parameters
        self.numberOfParticles = numberOfParticles
        self.worldLandmarks = worldLandmarks

        self.sigma_r1 = sigma_r1
        self.sigma_t = sigma_t
        self.sigma_r2 = sigma_r2

        self.sigma_range = sigma_range
        self.sigma_bearing = sigma_bearing

        # Initialize particles - x, y, heading, weight (uniform weight for initialization)
        sigma_x0, sigma_y0, sigma_theta0 = 2.0, 2.0, 0.1
        mu_x0, mu_y0, mu_theta0 = 0, 0, 0.1

        self.particles = np.concatenate((np.random.normal(mu_x0, sigma_x0, (self.numberOfParticles, 1)),
                                         np.random.normal(mu_y0, sigma_y0, (self.numberOfParticles, 1)),
                                         self.normalize_angles_array(np.random.normal(mu_theta0, sigma_theta0, (self.numberOfParticles, 1)))), axis=1)

        self.weights = (1/numberOfParticles) * np.ones(numberOfParticles)
        self.history = np.array((0, 0, 0.1)).reshape(1, 3)
        self.particles_history = np.expand_dims(self.particles.copy(), axis=0)

    def apply(self, Zt, Ut):
        """
        apply the particle filter on a single step in the sequence
        Parameters:
            Zt - the sensor measurement (range, bearing) as seen from the current position of the car
            Ut - the true odometry control command
        """

        # Motion model based on odometry
        self.motionModel(Ut)

        # Measurement prediction
        ParticlesLocation = self.MeasurementPrediction() # [range, bearing]

        # Sensor correction
        self.weightParticles(Zt, ParticlesLocation)

        # Estimate pose from best particle
        self.history = np.concatenate((self.history, self.bestKParticles(1).reshape(1, 3)), axis=0)

        # Resample particles
        self.resampleParticles()

        self.particles_history = np.concatenate((self.particles_history, np.expand_dims(self.particles.copy(), axis=0)), axis=0)

    def motionModel(self, odometry):
        """
        Apply the odometry motion model to the particles
        odometry - the true odometry control command
        the particles will be updated with the true odometry control command
        in addition, each particle will separately be added with Gaussian noise to its movement
        """
        dr1 = np.repeat(odometry['r1'], self.numberOfParticles) + np.random.normal(0, self.sigma_r1, self.numberOfParticles)
        dr1 = dr1.reshape(-1, 1)
        dt = np.repeat(odometry['t'], self.numberOfParticles) + np.random.normal(0, self.sigma_t, self.numberOfParticles)
        dt = dt.reshape(-1, 1)
        dr2 = np.repeat(odometry['r2'], self.numberOfParticles) + np.random.normal(0, self.sigma_r2, self.numberOfParticles)
        dr2 = dr2.reshape(-1, 1)
        theta = self.particles[:, 2].reshape(-1, 1)

        dMotion = np.concatenate((
            dt * np.cos(theta + dr1),
            dt * np.sin(theta + dr1),
            dr1 + dr2), axis=1)
        self.particles = self.particles + dMotion
        self.particles[:, 2] = self.normalize_angles_array(self.particles[:, 2])

    def MeasurementPrediction(self):
        """
        Calculates the measurement Prediction from the perspective of each of the particles
        returns: an array of size (number of particles x 2)
                 the first value is the range to the closest landmark and the second value is the bearing to it in radians

        """
        MeasurementPrediction = np.zeros((self.particles.shape[0], 2))  # range and bearing for each
        for i, particle in enumerate(self.particles):
            closest_landmark_idx =  np.argmin(np.linalg.norm(self.worldLandmarks - particle[0:2], axis=1)) # Size : NumLandmarks
            dist_xy =  self.worldLandmarks[closest_landmark_idx] - particle[0:2]
            r = np.linalg.norm(dist_xy)
            phi = np.arctan2(self.worldLandmarks[closest_landmark_idx,1] - particle[1], self.worldLandmarks[closest_landmark_idx,0]-particle[0]) - particle[2] # TODO (hint-differecne between the theta (landmark--particle) minus the heading of the particle)
            phi = ParticlesFilter.normalize_angle(phi)
            MeasurementPrediction[i, 0] = r
            MeasurementPrediction[i, 1] = phi
        return MeasurementPrediction

    def weightParticles(self, car_measurement, MeasurementPrediction):
        """
        Update the particle weights according to the normal Mahalanobis distance
        Parameters:
            car_measurement - the sensor measurement to the closet landmark (range, bearing) as seen from the position of the car
            MeasurementPredction - the Particles locations (range, bearing) related to the landmark
        """
        cov = np.cov((MeasurementPrediction - car_measurement).T) # TODO ( sensor measurements covariance matrix)
        for i, relatedLocations in enumerate(MeasurementPrediction):
            d = car_measurement - relatedLocations # TODO
            d[1] = ParticlesFilter.normalize_angle(d[1])
            Mahalanobis_distance_squared = np.dot(np.dot(d.T, np.linalg.inv(cov)), d) # TODO (hint- use the normal Mahalanobis distance)
            self.weights[i] = (1 / (2*np.pi*np.sqrt(np.linalg.det(cov)))) * np.exp(-0.5 * Mahalanobis_distance_squared) # TODO( hint: see normal distruntion , Multivariate Gaussian distributions)
        self.weights += 1.0e-200  # for numerical stability
        self.weights /= sum(self.weights)

    def resampleParticles(self):
        """
        law variance resampling
        """

        ind = []
        r = np.random.uniform() / self.numberOfParticles
        c = self.weights[0]  # weight of first particle
        i = 0
        for j in range(self.numberOfParticles):
            U = r + (j / self.numberOfParticles)
            while U > c:
                i += 1
                c += self.weights[i]
            ind.append(i)
        new_particles = self.particles[ind]

        self.particles = new_particles
        self.weights.resize(len(self.particles))
        self.weights.fill(1.0 / len(self.weights))

    def run(self, trueLandmarks, trueOdometry, trueTrajectory):
        """
        run the particle filter on a sequence of sensor measurements and odometry commands
        Parameters:
            odometry - a list of odometry commands
        """
        for i, timestamp in enumerate(range(0, trueOdometry.__len__() - 1)):

            # Observation model
            # calculate Zt - the range and bearing to the closest landmark as seen from the current true position of the robot
            closest_landmark_id = np.argmin(np.linalg.norm(trueLandmarks - trueTrajectory[i + 1, 0:2], axis=1)) # TODO (norma 1) hint (use, np.argmin,np.linalg.norm, trueLandmarks ,trueTrajectory[i + 1, 0:2])
            ClosetLandmarkLocation = trueLandmarks[closest_landmark_id] # TODO (hint use truelocation)
            dist_xy = trueLandmarks[closest_landmark_id] - trueTrajectory[i + 1, 0:2]
            r = np.linalg.norm(dist_xy) # TODO (norma 1)
            phi = np.arctan2(ClosetLandmarkLocation[1] - trueTrajectory[i+1, 1], ClosetLandmarkLocation[0] - trueTrajectory[i+1, 0]) - trueTrajectory[i+1, 2] # TODO
            r += float(np.random.normal(0, self.sigma_range, 1)) # TODO (add noise)
            phi += float(np.random.normal(0, self.sigma_bearing, 1)) # TODO (add noise)
            phi = self.normalize_angle(phi) # normalize (add noise)
            Zt = np.array([r, phi])

            self.apply(Zt, trueOdometry[timestamp])

            # if i % 10 == 0:
            #     num_particles = self.particles.shape[0]
            #     title = "pf_estimation_frame:{}_{}_particles".format(i, num_particles)
            #     graphs.draw_pf_frame(trueTrajectory, self.history, trueLandmarks, self.particles,
            #                   title.replace("_", " ").replace("pf", "Particle filter"))

    @staticmethod
    def normalize_angle(angle):
        """
        Normalize an angle to the range [-pi, pi]
        """
        while angle < -np.pi:
            angle += 2 * np.pi
        while angle >= np.pi:
            angle -= 2 * np.pi
        return angle

    @staticmethod
    def normalize_angles_array(angles):
        """
        applies normalize_angle on an array of angles
        """
        z = np.zeros_like(angles)
        for i in range(angles.shape[0]):
            z[i] = ParticlesFilter.normalize_angle(angles[i])
        return z

    def bestKParticles(self, K):
        """
        Given the particles and their weights, choose the top K particles according to the weights and return them
        """
        indexes = np.argsort(-self.weights) # TODO (use sort)
        bestK = indexes[:K]  # TODO (find the best K particles)
        return self.particles[bestK, :]

