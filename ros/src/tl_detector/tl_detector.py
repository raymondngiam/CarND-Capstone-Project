#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from light_classification.tl_classifier import TLClassifier
import tf
import tensorflow
import keras
import cv2
import yaml
import numpy as np

STATE_COUNT_THRESHOLD = 3
LOG_SECONDS = 3
SAVE_IMG_SECONDS = 2
PROCESS_IMG_INTERVAL = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.name = self.__class__.__name__

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.stop_line_positions = None
        self.light_stops = {} # Correlation between lights and stop positions
        self.correlated = False # Controls the correlation process
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.is_red_light = False

        self.image_cb_last_log_time = rospy.get_time()
        self.image_cb_last_img_time = rospy.get_time()
        self.traffic_cb_last_time = rospy.get_time()

        self.image_grab_count = 0
        self.num_classes = 2
        self.image_shape = (160, 576)

        self.tf_sess = tensorflow.Session()
    
        saver_path = "../../../model/model.ckpt.meta"
        saver = tensorflow.train.import_meta_graph(saver_path)
        saver.restore(self.tf_sess, "../../../model/model.ckpt")

        graph = tensorflow.get_default_graph()
        self.input_image = graph.get_tensor_by_name("input_image:0")
        self.keep_prob = graph.get_tensor_by_name("keep_prob:0")
        self.is_train = graph.get_tensor_by_name("is_train:0")
        self.logits = graph.get_tensor_by_name("logits:0")

        rospy.loginfo("%s: Model restored.", self.name)

        
        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, msg):
        if self.waypoints is None:
            self.waypoints = [wp.pose.pose.position for wp in msg.waypoints]

    def traffic_cb(self, msg):
        self.lights = msg.lights
        is_logging = False
        current_time = rospy.get_time()
        if (current_time - self.traffic_cb_last_time) > LOG_SECONDS:
           is_logging = True
           self.traffic_cb_last_time = current_time
    
        if self.waypoints == None:
            # Need the waypoints for further processing
            return
    
        # List of positions of the lines to stop in front of intersections
        self.stop_line_positions = self.config['stop_line_positions']
    
        # Associate the stop lines with the traffic lights. This is done once
        if self.correlated == False:
            self.correlate_lights_and_stop_positions()
            self.correlated = True

        # Get the closest waypoint to the position of the car
        if self.pose:
            car_wp_index = self.get_closest_waypoint(self.pose.pose.position.x,
                                                     self.pose.pose.position.y)
            if (is_logging):
                rospy.loginfo("%s: car @ index [%s]", self.name, car_wp_index)
                rospy.loginfo("%s: car @ coord [%s, %s]", self.name, \
                                self.pose.pose.position.x, \
                                self.pose.pose.position.y)
        else:
            # Cannot continue without knowing the pose of the car itself
            return

        # Locate the next upcoming red traffic light stop line waypoint index
        closest_stop_index = len(self.waypoints) - 1

        stop_wp_indices = []
        stop_wp_dict = dict()

        for i,light in enumerate(self.lights):
                
            # Get the stop line from the light
            light_x = light.pose.pose.position.x
            light_y = light.pose.pose.position.y
            stop_line = self.light_stops[(light_x, light_y)]

            # Get the waypoint index closest to the stop line
            stop_line_x = stop_line[0]
            stop_line_y = stop_line[1]
            stop_wp_index = self.get_closest_waypoint(stop_line_x, \
                                                          stop_line_y)

            # Store
            stop_wp_indices.append(stop_wp_index)
            stop_wp_dict[i] = stop_wp_index

            if (is_logging):
                rospy.loginfo("%s: Traffic light %s @ %s", self.name, \
                              i+1, \
                              stop_wp_index)

        behind_light_stop_wp = [stop_idx > car_wp_index for stop_idx in stop_wp_indices]
        next_idx = np.argmax(behind_light_stop_wp)
        self.next_light_stop_wp = stop_wp_dict[next_idx]
        if (is_logging):
            rospy.loginfo("%s: closest stop @ %s", self.name, self.next_light_stop_wp)

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        current_time = rospy.get_time()
        is_logging = False
        if (current_time - self.image_cb_last_log_time > LOG_SECONDS):
            is_logging = True
            self.image_cb_last_log_time = current_time
        is_saving_img = False
        if (current_time - self.image_cb_last_img_time > SAVE_IMG_SECONDS):
            is_saving_img = True
            self.image_cb_last_img_time = current_time

        self.has_image = True
        self.camera_image = msg

        if (is_logging):
            rospy.logwarn("%s: len(waypoint): %s", self.name, len(self.waypoints))

        is_process_img = True
        self.image_grab_count += 1
        if self.image_grab_count > PROCESS_IMG_INTERVAL:
            is_process_img = True
            self.image_grab_count = 0
            #light_wp, state = self.process_traffic_lights()

        car_wp_index = -1
        if self.pose:
            car_wp_index = self.get_closest_waypoint(self.pose.pose.position.x,
                                                     self.pose.pose.position.y)

        self.is_red_light = False
        if (is_process_img and car_wp_index != -1 and \
                (self.next_light_stop_wp - car_wp_index) > 0 and \
                (self.next_light_stop_wp - car_wp_index) < 150):
            input_image = self.input_image
            keep_prob = self.keep_prob
            is_train = self.is_train

            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            width=576
            height=160
            frame_resized = cv2.resize(frame,(width, height), interpolation = cv2.INTER_NEAREST)
            cv_image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            image = np.reshape(cv_image, (-1,height,width,3))
            prob_classes = self.tf_sess.run(tensorflow.nn.softmax(self.logits), 
                                            feed_dict={
                                                input_image: image,
                                                keep_prob: 1.0,
                                                is_train: False
                                            })
            prob_classes = prob_classes.reshape((-1,3))
            rospy.logwarn("%s: prob_classes: %s", self.name, prob_classes)
            index_result = np.argmax(prob_classes, axis=1)
            one_hot_result = np.zeros_like(prob_classes)
            one_hot_result[np.arange(index_result.shape[0]), index_result] = 1
            rospy.logwarn("%s: one-hot result: %s", self.name, one_hot_result)

            self.is_red_light = index_result[0] == 0
            rospy.logwarn("%s: is_red_light : %s", self.name, self.is_red_light)

            if (is_saving_img):
                try:
                    cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                    file_name = "../../../imgs/tl_detector/" + str(current_time) + "{}".format(one_hot_result[0]) + ".jpg"
                    cv2.imwrite(file_name, cv_image)
                except CvBridgeError as e:
                    rospy.logerr("%s: CvBridgeError: %s", self.name, e)

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        #if self.state != state:
        #    self.state_count = 0
        #    self.state = state
        #elif self.state_count >= STATE_COUNT_THRESHOLD:
        #    self.last_state = self.state
        #    light_wp = light_wp if state == TrafficLight.RED else -1
        #    self.last_wp = light_wp
        #    self.upcoming_red_light_pub.publish(Int32(light_wp))
        #else:
        #    self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        #self.state_count += 1
        if (self.is_red_light):
            self.upcoming_red_light_pub.publish(self.next_light_stop_wp)
        else:
            self.upcoming_red_light_pub.publish(len(self.waypoints) - 1)

    def correlate_lights_and_stop_positions(self):
        """
        Assign the closest stop line position to each of the traffic lights.
        The operation is supposed to be done only once
        """
        for light in self.lights:
            # Reset the minimum distance and the index we search for
            min_dist = float('inf')
            matching_index = 0
            
            for i, stop_line_position in enumerate(self.stop_line_positions):
                # Calculate the Euclidean distance
                dx = light.pose.pose.position.x - stop_line_position[0]
                dy = light.pose.pose.position.y - stop_line_position[1]
                dist = pow(dx, 2) + pow(dy, 2)
                
                # Update the minimum distance and matching index
                if dist < min_dist:
                    min_dist = dist
                    matching_index = i
        
            # Correlate each light position (x, y) with the closest stop line
            x = light.pose.pose.position.x
            y = light.pose.pose.position.y
            self.light_stops[(x, y)] = self.stop_line_positions[matching_index]

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        min_dist = float('inf')
        closest_waypoint_index = 0  # Index to return

        for i, wp in enumerate(self.waypoints):
            # d^2 = (x1 - x2)^2 + (y1 - y2)^2
            dist = pow(x - wp.x, 2) + pow(y - wp.y, 2)
            
            # Update the minimum distance and update the index
            if dist < min_dist:
                min_dist = dist
                closest_waypoint_index = i
    
        # Return the index of the closest waypoint in self.waypoints
        return closest_waypoint_index

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

        #TODO find the closest visible traffic light (if one exists)

        if light:
            state = self.get_light_state(light)
            return light_wp, state
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
